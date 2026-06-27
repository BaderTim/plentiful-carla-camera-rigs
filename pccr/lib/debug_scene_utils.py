"""
Live CARLA scene helpers for interactive debug tools.

These utilities manage Traffic Manager autopilot spawning and actor lifecycle
for the annotation debug tool (``debug/annotation_debug_tool.py``).  They are
**not** used by the main data-capture pipeline (``core/``), which replays
pre-recorded trajectories instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import carla


@dataclass
class WalkerSpawnResult:
    """Bundles walkers, their AI controllers, and spawn metadata.

    Attributes:
        walkers: Spawned walker actors.
        controllers: Matching AI walker controller actors.
        blueprint_ids: Blueprint ID string for each spawned walker.
        spawn_transforms: World transform used to spawn each walker.
    """

    walkers: List[carla.Actor] = field(default_factory=list)
    controllers: List[carla.Actor] = field(default_factory=list)
    blueprint_ids: List[str] = field(default_factory=list)
    spawn_transforms: List[carla.Transform] = field(default_factory=list)


def configure_traffic_manager(
    client: carla.Client,
    port: int = 8000,
    seed: int = 42,
    synchronous_mode: bool = True,
) -> carla.TrafficManager:
    """Configure a Traffic Manager instance for deterministic autopilot scenes.

    Args:
        client: Connected CARLA client.
        port: Traffic Manager RPC port.
        seed: Random seed for reproducible behaviour.
        synchronous_mode: When ``True`` TM steps in lock-step with the world tick.

    Returns:
        Configured :class:`carla.TrafficManager` instance.
    """
    tm = client.get_trafficmanager(port)
    tm.set_synchronous_mode(synchronous_mode)
    tm.set_random_device_seed(seed)
    tm.set_hybrid_physics_mode(False)
    tm.set_global_distance_to_leading_vehicle(2.0)
    return tm


def spawn_traffic(
    world: carla.World,
    traffic_density: float,
    ego_spawn_point: carla.Transform,
    seed: int = 42,
) -> Tuple[List[carla.Actor], List[str], List[carla.Transform]]:
    """Spawn traffic vehicles via autopilot away from the ego spawn point.

    Args:
        world: CARLA world.
        traffic_density: Fraction of available spawn points to fill (0–1) or
            absolute vehicle count (> 1).
        ego_spawn_point: Ego vehicle spawn transform; vehicles will be placed at
            least 10 m away from this point.
        seed: RNG seed for reproducible spawning.

    Returns:
        A three-tuple of:

        * ``traffic_vehicles`` — spawned vehicle actors.
        * ``blueprint_ids`` — blueprint ID string for each spawned vehicle.
        * ``spawn_transforms`` — world transforms used to spawn each vehicle.
    """
    rng = random.Random(seed)
    traffic_vehicles: List[carla.Actor] = []
    blueprint_ids: List[str] = []
    spawn_transforms: List[carla.Transform] = []

    all_spawn_points = world.get_map().get_spawn_points()
    all_spawn_points.sort(key=lambda sp: (sp.location.x, sp.location.y, sp.location.z))

    available = [
        sp
        for sp in all_spawn_points
        if ego_spawn_point.location.distance(sp.location) > 10.0
    ]
    if not available:
        return traffic_vehicles, blueprint_ids, spawn_transforms

    if traffic_density <= 1.0:
        num_vehicles = int(traffic_density * len(available))
    else:
        num_vehicles = int(traffic_density)
    num_vehicles = max(0, min(num_vehicles, len(available)))
    if num_vehicles == 0:
        return traffic_vehicles, blueprint_ids, spawn_transforms

    selected = rng.sample(available, num_vehicles)
    selected.sort(key=lambda sp: (sp.location.x, sp.location.y, sp.location.z))

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = sorted(blueprint_library.filter("vehicle.*"), key=lambda bp: bp.id)
    if not vehicle_blueprints:
        return traffic_vehicles, blueprint_ids, spawn_transforms

    for sp in selected:
        vehicle_bp = vehicle_blueprints[rng.randint(0, len(vehicle_blueprints) - 1)]
        try:
            vehicle = world.spawn_actor(vehicle_bp, sp)
        except RuntimeError:
            continue
        traffic_vehicles.append(vehicle)
        blueprint_ids.append(vehicle_bp.id)
        spawn_transforms.append(sp)

    return traffic_vehicles, blueprint_ids, spawn_transforms


def spawn_pedestrians(
    client: carla.Client,
    world: carla.World,
    max_pedestrians: int = 50,
    seed: int = 42,
) -> WalkerSpawnResult:
    """Spawn walkers using the CARLA navigation mesh and attach AI controllers.

    Args:
        client: Connected CARLA client (used for batch spawning).
        world: CARLA world.
        max_pedestrians: Upper bound on spawned walker count.
        seed: RNG seed for reproducible spawning.

    Returns:
        :class:`WalkerSpawnResult` with walkers and their controllers.
    """
    result = WalkerSpawnResult()
    if max_pedestrians <= 0:
        return result

    rng = random.Random(seed)
    blueprint_library = world.get_blueprint_library()
    walker_bps = sorted(blueprint_library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
    walker_controller_bp = blueprint_library.find("controller.ai.walker")

    if not walker_bps or walker_controller_bp is None:
        return result

    # Collect candidate spawn locations.
    spawn_points: List[carla.Transform] = []
    for _ in range(max_pedestrians * 2):
        location = world.get_random_location_from_navigation()
        if location is not None:
            spawn_points.append(carla.Transform(location=location))

    if not spawn_points:
        return result

    spawn_points.sort(key=lambda sp: (sp.location.x, sp.location.y, sp.location.z))
    spawn_points = spawn_points[:max_pedestrians]

    bp_indices = [rng.randint(0, len(walker_bps) - 1) for _ in spawn_points]
    speeds = [1.0 + rng.random() * 1.5 for _ in spawn_points]

    walker_batch = []
    for idx, sp in enumerate(spawn_points):
        walker_bp = walker_bps[bp_indices[idx]]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker_batch.append(carla.command.SpawnActor(walker_bp, sp))

    spawn_results = client.apply_batch_sync(walker_batch, True)
    spawned_speed_indices: List[int] = []
    for idx, res in enumerate(spawn_results):
        if res.error:
            continue
        walker = world.get_actor(res.actor_id)
        if walker is None:
            continue
        result.walkers.append(walker)
        result.blueprint_ids.append(walker_bps[bp_indices[idx]].id)
        result.spawn_transforms.append(spawn_points[idx])
        spawned_speed_indices.append(idx)

    if not result.walkers:
        return result

    world.tick()

    controller_batch = [
        carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker)
        for walker in result.walkers
    ]
    for ctrl_res in client.apply_batch_sync(controller_batch, True):
        if ctrl_res.error:
            continue
        ctrl = world.get_actor(ctrl_res.actor_id)
        if ctrl is not None:
            result.controllers.append(ctrl)

    if not result.controllers:
        return result

    world.tick()

    for idx, ctrl in enumerate(result.controllers):
        try:
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            speed_idx = spawned_speed_indices[idx] if idx < len(spawned_speed_indices) else None
            ctrl.set_max_speed(speeds[speed_idx] if speed_idx is not None else 1.5)
        except RuntimeError:
            continue

    return result


def enable_autopilot(
    ego_vehicle: carla.Actor, traffic_vehicles: List[carla.Actor], tm_port: int
) -> None:
    """Enable Traffic Manager autopilot on the ego vehicle and traffic vehicles.

    Args:
        ego_vehicle: Ego vehicle actor.
        traffic_vehicles: Spawned traffic vehicle actors.
        tm_port: Traffic Manager RPC port.
    """
    ego_vehicle.set_autopilot(True, tm_port)
    for vehicle in traffic_vehicles:
        try:
            vehicle.set_autopilot(True, tm_port)
        except Exception:
            continue


def destroy_actors(actors: List[carla.Actor]) -> None:
    """Destroy a list of CARLA actors, silently ignoring already-destroyed ones.

    Args:
        actors: Actors to destroy (``None`` entries are skipped).
    """
    for actor in actors:
        if actor is None:
            continue
        try:
            actor.destroy()
        except Exception:
            continue


def stop_and_destroy_walkers(result: WalkerSpawnResult) -> None:
    """Stop walker AI controllers then destroy both controllers and walkers.

    Args:
        result: :class:`WalkerSpawnResult` from :func:`spawn_pedestrians`.
    """
    for ctrl in result.controllers:
        try:
            ctrl.stop()
        except Exception:
            continue
    destroy_actors(result.controllers)
    destroy_actors(result.walkers)
