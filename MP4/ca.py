
import math

numARows = 46
numAColumns = 260
numBRows = 260
numBColumns = 914
numCRows = 46 
numCColumns = 914
tileSize = 32


# Calculate the number of tiles in each dimension
tiles_in_row = math.ceil(numCRows / 32)
tiles_in_column = math.ceil(numCColumns / 32)

# Calculate the total number of tiles
total_tiles = tiles_in_row * tiles_in_column

# Calculate the total number of floating-point operations
total_ops = total_tiles * (32 * 32 * (2 * numAColumns - 1))

print (total_ops)
