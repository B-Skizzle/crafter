import functools

import numpy as np
import opensimplex

from . import constants
from . import objects

def parse_ascii_map(map_string):
  """
  Parse an ASCII map string into terrain and entity positions.

  Returns:
    - terrain_grid: 2D list of material names
    - entities: list of (entity_type, x, y) tuples
    - player_start: (x, y) tuple or None
  """
  lines = map_string.strip().split('\n')
  height = len(lines)
  width = max(len(line) for line in lines)

  # Initialize terrain grid with grass as default
  terrain_grid = [['grass' for _ in range(width)] for _ in range(height)]
  entities = []
  player_start = None

  # Character to terrain mapping
  terrain_map = {
    '*': 'mine',
    '|': 'grass',
    'o': 'stone',
    '~': 'water',
    '.': 'sand',
    '@': 'water',  
    'E': 'water', 
    'S': 'water',
  }

  for y, line in enumerate(lines):
    for x, char in enumerate(line):
      # Set terrain
      if char in terrain_map:
        terrain_grid[y][x] = terrain_map[char]

      # Mark entities
      if char == '@':
        player_start = (x, y)
      elif char == 'E':
        entities.append(('EnemyBoat', x, y))
      elif char == '*':
        entities.append(('Mine', x, y))

  return terrain_grid, entities, player_start

def generate_world(world, player, enemy_config=None, ascii_map=None):
  """
  Generate world either from ASCII map or procedurally.
  
  Args:
    world: World object to populate
    player: Player object
    enemy_config: Enemy spawn configuration
    ascii_map: Optional ASCII map string. If provided, terrain is loaded from map.
  """
  # Use default config if none provided
  if enemy_config is None:
    enemy_config = {
      'enemies_enabled': True,
      'zombie_settings': {'enabled': True},
      'skeleton_settings': {'enabled': True}
    }
  
  # TERRAIN GENERATION
  if ascii_map:
    # Load terrain from ASCII map
    terrain_grid, manual_entities, player_start = parse_ascii_map(ascii_map)
    
    # Apply terrain to world
    for y in range(len(terrain_grid)):
      for x in range(len(terrain_grid[y])):
        if x < world.area[0] and y < world.area[1]:
          world[x, y] = terrain_grid[y][x]
    
    # Move player to starting position if specified
    if player_start:
      world.move(player, player_start)  # Proper movement with chunk tracking!
    
    # Place manually specified entities
    for entity_type, x, y in manual_entities:
      if entity_type == 'Cow':
        world.add(objects.Cow(world, (x, y)))
      elif entity_type == 'Zombie':
        world.add(objects.Zombie(world, (x, y), player))
      elif entity_type == 'EnemyBoat':
        world.add(objects.EnemyBoat(world, (x, y), player))
    
  else:
    # Original procedural generation
    simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
    tunnels = np.zeros(world.area, bool)
    for x in range(world.area[0]):
      for y in range(world.area[1]):
        _set_material(world, (x, y), player, tunnels, simplex)
  
  # ENTITY SPAWNING (always runs, even with ASCII maps)
  # This allows procedural spawning on top of loaded terrain
  if not ascii_map:  # Only do procedural spawning if no manual entities
    tunnels = np.zeros(world.area, bool) if ascii_map else tunnels
    for x in range(world.area[0]):
      for y in range(world.area[1]):
        _set_object(world, (x, y), player, tunnels, enemy_config)



def _set_material(world, pos, player, tunnels, simplex):
  x, y = pos
  simplex = functools.partial(_simplex, simplex)
  uniform = world.random.uniform
  start = 4 - np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
  start += 2 * simplex(x, y, 8, 3)
  start = 1 / (1 + np.exp(-start))
  grass = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
  grass -= 2 * start
  mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
  mountain -= 4 * start + 0.3 * grass
  if start > 0.5:
    world[x, y] = 'water'
  elif mountain > 0.15:
    if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
      world[x, y] = 'path'
    elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
      world[x, y] = 'path'
      tunnels[x, y] = True
    elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
      world[x, y] = 'path'
      tunnels[x, y] = True
    elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
      world[x, y] = 'coal'
    elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
      world[x, y] = 'iron'
    elif mountain > 0.18 and uniform() > 0.994:
      world[x, y] = 'diamond'
    elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35:
      world[x, y] = 'lava'
    else:
      world[x, y] = 'stone'
  elif 0.25 < grass <= 0.35 and simplex(x, y, 4, 9) > -0.2:
    world[x, y] = 'sand'
  elif 0.3 < grass:
    world[x, y] = 'grass'
  else:  # waterland
    if simplex(x, y, 5, 7) > 0 and uniform() > 0.9:
      world[x, y] = 'mine'
    else:
      world[x, y] = 'water'


def _set_object(world, pos, player, tunnels, enemy_config):
  x, y = pos
  uniform = world.random.uniform
  dist = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
  material, _ = world[x, y]

  if material not in constants.walkable:
    pass
  elif dist > 3 and material == 'water' and uniform() > 0.985:
    world.add(objects.Cow(world, (x, y)))
  # Only spawn enemies if enabled in config
  elif enemy_config['enemies_enabled']:
    # Zombie spawning
    if enemy_config['zombie_settings']['enabled']:
      zombie_rate = enemy_config['zombie_settings'].get('worldgen_spawn_probability', 0.007)
      if dist > 10 and uniform() < zombie_rate:
        world.add(objects.Zombie(world, (x, y), player))
        return  # Don't spawn multiple enemies at same location
    # Enemy Boat spawning
    if enemy_config['skeleton_settings']['enabled']:
      skeleton_rate = enemy_config['skeleton_settings'].get('worldgen_spawn_probability', 0.05)
      if material == 'path' and tunnels[x, y] and uniform() < skeleton_rate:
        world.add(objects.EnemyBoat(world, (x, y), player))


def _simplex(simplex, x, y, z, sizes, normalize=True):
  if not isinstance(sizes, dict):
    sizes = {sizes: 1}
  value = 0
  for size, weight in sizes.items():
    if hasattr(simplex, 'noise3d'):
      noise = simplex.noise3d(x / size, y / size, z)
    else:
      noise = simplex.noise3(x / size, y / size, z)
    value += weight * noise
  if normalize:
    value /= sum(sizes.values())
  return value
