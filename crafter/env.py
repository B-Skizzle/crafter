import collections
import os
import numpy as np
import json 

from . import constants
from . import engine
from . import objects
from . import worldgen


# Gym is an optional dependency.
try:
  import gym
  DiscreteSpace = gym.spaces.Discrete
  BoxSpace = gym.spaces.Box
  DictSpace = gym.spaces.Dict
  BaseClass = gym.Env
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object


class Env(BaseClass):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, seed=None, config_file='enemy_config.json',
      map_file=None):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
    self._area = area
    self._view = view
    self._size = size
    self._reward = reward
    self._length = length
    self._seed = seed
    
    # Load enemy configuration from JSON file
    config_path = constants.root / config_file  # Find the config file
    if os.path.exists(config_path):  # Check if the file exists
        with open(config_path, 'r') as f:  # Open the file for reading
            self._enemy_config = json.load(f)  # Read and convert JSON to a Python dictionary
    else:
        # If no config file exists, use default settings
        self._enemy_config = {
            'enemies_enabled': True,
            'spawn_count_multiplier': 1.0,
            'spawn_frequency_multiplier': 1.0,
            'zombie_settings': {
                'enabled': True,
                'spawn_probability': 0.3,
                'despawn_probability': 0.4,
                'min_count': 3.5,
                'max_count': 3.5
            },
            'skeleton_settings': {
                'enabled': True,
                'spawn_probability': 0.1,
                'despawn_probability': 0.1,
                'min_count': 1,
                'max_count': 2
            }
        }

    # Load ASCII map if provided
    self._ascii_map = None
    if map_file:
      map_path = constants.root / map_file
      if os.path.exists(map_path):
        with open(map_path, 'r') as f:
          self._ascii_map = f.read()
        lines = self._ascii_map.strip().split('\n')
        map_height = len(lines)
        map_width = max(len(line) for line in lines)

        # Override area with map dimensions
        area = (map_width, map_height)
        self._area = area
        print(f"Loaded map '{map_file}' with dimensions: {map_width}×{map_height}")
        
      else:
        print(f"Warning: Map file '{map_file}' not found. Using procedural generation.")

    self._episode = 0
    self._world = engine.World(area, constants.materials, (12, 12))
    self._textures = engine.Textures(constants.root / 'assets')
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._local_view = engine.LocalView(
        self._world, self._textures, [view[0], view[1] - item_rows])
    self._item_view = engine.ItemView(
        self._textures, [view[0], item_rows])
    self._sem_view = engine.SemanticView(self._world, [
        objects.Player, objects.Cow, objects.Zombie,
        objects.EnemyBoat, objects.Missile, objects.Plant])
    self._step = None
    self._player = None
    self._last_health = None
    self._unlocked = None
    # Some libraries expect these attributes to be set.
    self.reward_range = None
    self.metadata = None

  @property
  def observation_space(self):
    return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

  @property
  def action_space(self):
    return DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def reset(self):
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    self._episode += 1
    self._step = 0
    self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._update_time()
    self._player = objects.Player(self._world, center)
    self._last_health = self._player.health
    self._world.add(self._player)
    self._unlocked = set()
    worldgen.generate_world(
      self._world,
      self._player,
      self._enemy_config,
      ascii_map=self._ascii_map,
    )
    return self._obs()

  def step(self, action):
    self._step += 1
    self._update_time()

    # Check if player is dead and in death delay
    if self._player.death_delay > 0:
      self._player.death_delay -= 1  # Count down death delay
      self._player.action = 'noop'  # Force player to do nothing during death
      # Don't update world objects during death delay - freeze the game
    # NEW: Check if player has a cooldown timer running
    elif self._player.movement_cooldown > 0:
      self._player.movement_cooldown -= 1  # Count down by 1
      self._player.action = 'noop'  # Force player to do nothing during cooldown
      # Skip player action but still update world
      for obj in self._world.objects:
        if self._player.distance(obj) < 2 * max(self._view):
          obj.update()
    else:
      # Normal action processing (this was the old code)
      self._player.action = constants.actions[action]
      for obj in self._world.objects:
        if self._player.distance(obj) < 2 * max(self._view):
          obj.update()


        
    if self._step % 10 == 0:
      for chunk, objs in self._world.chunks.items():
        # xmin, xmax, ymin, ymax = chunk
        # center = (xmax - xmin) // 2, (ymax - ymin) // 2
        # if self._player.distance(center) < 4 * max(self._view):
        self._balance_chunk(chunk, objs)
    obs = self._obs()
    reward = (self._player.health - self._last_health) / 10
    self._last_health = self._player.health
    unlocked = {
        name for name, count in self._player.achievements.items()
        if count > 0 and name not in self._unlocked}
    if unlocked:
      self._unlocked |= unlocked
      reward += 1.0
    dead = self._player.health <= 0
    over = self._length and self._step >= self._length
    done = dead or over
    info = {
        'inventory': self._player.inventory.copy(),
        'achievements': self._player.achievements.copy(),
        'discount': 1 - float(dead),
        'semantic': self._sem_view(),
        'player_pos': self._player.pos,
        'reward': reward,
    }
    if not self._reward:
      reward = 0.0
    return obs, reward, done, info

  def render(self, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    local_view = self._local_view(self._player, unit)
    item_view = self._item_view(self._player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))
  
  def save_game(self, filepath):
    """Save the entire game state to a file"""
    
    # Save the World (the ground/terrain map)
    # A snapshot of the entire world from above
    world_data = {
        'mat_map': self._world._mat_map,  # This is the terrain (grass, water, stone, etc.)
        'area': self._world.area,  # How big the world is (64x64)
        'daylight': self._world.daylight,  # Is it day or night?
    }
    
    # Save the Player state
    # Saves where the player is, what they have, and their stats
    player_data = {
        'pos': self._player.pos,  # (x, y coordinates)
        'facing': self._player.facing,  # Which direction you're looking
        'inventory': self._player.inventory.copy(),  # Everything in your backpack
        'achievements': self._player.achievements.copy(),  # What you've unlocked
        'health': self._player.health,  # How much health you have
        'sleeping': self._player.sleeping,  # Are you asleep?
        'hunger': self._player._hunger,  # Internal hunger timer
        'thirst': self._player._thirst,  # Internal thirst timer
        'fatigue': self._player._fatigue,  # Internal tiredness timer
        'recover': self._player._recover,  # Internal recovery timer
        'movement_cooldown': self._player.movement_cooldown,  # Movement timer
        'death_delay': self._player.death_delay,  # Death explosion delay
    }
    
    # Save all the Objects (cows, zombies, plants, etc.)
    # In a list, we're keeping track of:
    # - What type of thing it is
    # - Where it is
    # - Its health and other details
    objects_data = []
    for obj in self._world.objects:
        if obj == self._player:  # Skip the player (we saved them already!)
            continue
            
        # Create a dictionary (like a form) for this object
        obj_dict = {
            'type': type(obj).__name__,  # Is it a Cow, Zombie, Plant, etc.?
            'pos': obj.pos.tolist(),  # Where is it? (.tolist() makes it saveable)
            'health': obj.health,  # How much health?
        }
        
        # Add special info for specific types
        if type(obj).__name__ == 'Zombie':
            obj_dict['cooldown'] = obj.cooldown  # Attack timer
        elif type(obj).__name__ == 'EnemyBoat':
            obj_dict['reload'] = obj.reload  # Missile reload timer
            obj_dict['facing'] = obj.facing  # Which direction it's facing
        elif type(obj).__name__ == 'Missile':
            obj_dict['facing'] = obj.facing  # Which way it's flying
        elif type(obj).__name__ == 'Plant':
            obj_dict['grown'] = obj.grown  # How much it's grown
            
        objects_data.append(obj_dict)
    
    # Save Environment info (episode number, time step, etc.)
    env_data = {
        'step': self._step,  # How many game ticks have happened
        'episode': self._episode,  # Which episode number
        'seed': self._seed,  # The random seed (for world generation)
        'last_health': self._last_health,  # Player's health last frame
    }
    
    # Bundle everything together and save to a file
    # np.savez saves everything in a compressed file (like a zip file for game data)
    np.savez_compressed(
        filepath,
        # World data
        world_mat_map=world_data['mat_map'],
        world_area=world_data['area'],
        world_daylight=world_data['daylight'],
        # Player data
        player_pos=player_data['pos'],
        player_facing=player_data['facing'],
        player_inventory=player_data['inventory'],
        player_achievements=player_data['achievements'],
        player_sleeping=player_data['sleeping'],
        player_hunger=player_data['hunger'],
        player_thirst=player_data['thirst'],
        player_fatigue=player_data['fatigue'],
        player_recover=player_data['recover'],
        player_movement_cooldown=player_data['movement_cooldown'],
        player_death_delay=player_data['death_delay'],
        # Objects
        objects=objects_data,
        # Environment
        env_step=env_data['step'],
        env_episode=env_data['episode'],
        env_seed=env_data['seed'],
        env_last_health=env_data['last_health'],
    )
    
    print(f"Game saved to {filepath}")
  
  def load_game(self, filepath):
    """Load a saved game from a file"""
    
    # Step 1: Open the save file and read everything
    # np.load is like unzipping the backpack we saved earlier
    data = np.load(filepath, allow_pickle=True)
    
    # Step 2: Restore the World
    # Clear everything first (like erasing the chalkboard)
    self._world._mat_map = data['world_mat_map']
    self._world.area = tuple(data['world_area'])
    self._world.daylight = float(data['world_daylight'])
    
    # Reset the object lists (we'll fill them back up)
    self._world._objects = [None]  # Start with empty list (None at index 0)
    self._world._chunks = collections.defaultdict(set)  # Empty chunks
    self._world._obj_map = np.zeros(self._world.area, np.uint32)  # Clear object positions
    
    # Step 3: Restore the Player
    # Create the player at the saved position
    center = tuple(data['player_pos'])
    self._player = objects.Player(self._world, center)
    
    # Load all the player's stuff
    self._player.facing = tuple(data['player_facing'])
    self._player.inventory = data['player_inventory'].item()  # .item() unpacks it
    self._player.achievements = data['player_achievements'].item()
    self._player.sleeping = bool(data['player_sleeping'])
    self._player._hunger = float(data['player_hunger'])
    self._player._thirst = float(data['player_thirst'])
    self._player._fatigue = float(data['player_fatigue'])
    self._player._recover = float(data['player_recover'])
    self._player.movement_cooldown = int(data['player_movement_cooldown'])
    self._player.death_delay = int(data['player_death_delay'])

    # Add the player to the world
    self._world.add(self._player)
    
    # Step 4: Restore all Objects (cows, zombies, plants, etc.)
    # Loop through the list of saved objects
    for obj_data in data['objects']:
        obj_type = obj_data['type']  # What type? (Cow, Zombie, etc.)
        pos = tuple(obj_data['pos'])  # Where is it?
        
        # Create the right type of object
        if obj_type == 'Cow':
            obj = objects.Cow(self._world, pos)
        elif obj_type == 'Zombie':
            obj = objects.Zombie(self._world, pos, self._player)
            obj.cooldown = obj_data['cooldown']
        elif obj_type == 'EnemyBoat':
            obj = objects.EnemyBoat(self._world, pos, self._player)
            obj.reload = obj_data['reload']
            obj.facing = tuple(obj_data['facing'])
        elif obj_type == 'Missile':
            obj = objects.Missile(self._world, pos, tuple(obj_data['facing']))
        elif obj_type == 'Plant':
            obj = objects.Plant(self._world, pos)
            obj.grown = obj_data['grown']
        elif obj_type == 'Fence':
            obj = objects.Fence(self._world, pos)
        else:
            continue  # Skip unknown types
        
        # Set the health
        obj.health = obj_data['health']
        
        # Add it to the world
        self._world.add(obj)
    
    # Restore Environment state
    self._step = int(data['env_step'])
    self._episode = int(data['env_episode'])
    self._seed = int(data['env_seed'])
    self._last_health = float(data['env_last_health'])
    
    print(f"Game loaded from {filepath}")
    
    # Return the current view (what the player sees)
    return self._obs()
  
  
  def _obs(self):
    return self.render()

  def _update_time(self):
    # https://www.desmos.com/calculator/grfbc6rs3h
    progress = (self._step / 300) % 1 + 0.3
    daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    self._world.daylight = daylight

  def _balance_chunk(self, chunk, objs):
      light = self._world.daylight

      
      # Only spawn enemies if enabled in config
      if self._enemy_config['enemies_enabled']:
          # Zombies
          if self._enemy_config['zombie_settings']['enabled']:
              z_cfg = self._enemy_config['zombie_settings']
              spawn_mult = self._enemy_config['spawn_count_multiplier']
              freq_mult = self._enemy_config['spawn_frequency_multiplier']

              self._balance_object(
                  chunk, objs, objects.Zombie, 'water', 6, 0,
                  z_cfg['spawn_probability'] * freq_mult,
                  z_cfg['despawn_probability'] * freq_mult,
                  lambda pos: objects.Zombie(self._world, pos, self._player),
                  lambda num, space: (
                      0 if space < 50 else z_cfg['min_count'] * spawn_mult - 3 * light,
                      z_cfg['max_count'] * spawn_mult - 3 * light))

          # Enemy Boats
          if self._enemy_config['skeleton_settings']['enabled']:
              s_cfg = self._enemy_config['skeleton_settings']
              spawn_mult = self._enemy_config['spawn_count_multiplier']
              freq_mult = self._enemy_config['spawn_frequency_multiplier']

              self._balance_object(
                  chunk, objs, objects.EnemyBoat, 'path', 7, 7,
                  s_cfg['spawn_probability'] * freq_mult,
                  s_cfg['despawn_probability'] * freq_mult,
                  lambda pos: objects.EnemyBoat(self._world, pos, self._player),
                  lambda num, space: (
                      0 if space < 6 else s_cfg['min_count'] * spawn_mult,
                      s_cfg['max_count'] * spawn_mult))

      
      self._balance_object(
          chunk, objs, objects.Cow, 'water', 5, 5, 0.01, 0.1,
          lambda pos: objects.Cow(self._world, pos),
          lambda num, space: (0 if space < 30 else 1, 1.5 + light))


  def _balance_object(
      self, chunk, objs, cls, material, span_dist, despan_dist,
      spawn_prob, despawn_prob, ctor, target_fn):
    xmin, xmax, ymin, ymax = chunk
    random = self._world.random
    creatures = [obj for obj in objs if isinstance(obj, cls)]
    mask = self._world.mask(*chunk, material)
    target_min, target_max = target_fn(len(creatures), mask.sum())
    if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
      xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
      ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
      xs, ys = xs[mask], ys[mask]
      i = random.randint(0, len(xs))
      pos = np.array((xs[i], ys[i]))
      empty = self._world[pos][1] is None
      away = self._player.distance(pos) >= span_dist
      if empty and away:
        self._world.add(ctor(pos))
    elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
      obj = creatures[random.randint(0, len(creatures))]
      away = self._player.distance(obj.pos) >= despan_dist
      if away:
        self._world.remove(obj)
