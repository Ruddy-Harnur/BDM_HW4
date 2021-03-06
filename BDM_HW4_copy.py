import csv
import sys
from pyspark import SparkContext
import fiona
import fiona.crs
import shapely
import rtree
import geopandas as gpd


# This is to load the shape file
shapefile = sys.argv[1]

# And project it into EPSG:2263 (NAD 83 NY State) plane
neighborhoods = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))


def createIndex(shapefile):
    '''
    This function takes in a shapefile path, and return:
    (1) index: an R-Tree based on the geometry data in the file
    (2) zones: the original data of the shapefile
    
    Note that the ID used in the R-tree 'index' is the same as
    the order of the object in zones.
    '''
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)
    
    
def findPickupZone(p, index, zones):
    '''
    findZone returned the ID of the shape (stored in 'zones' with
    'index') that contains the given point 'p'. If there's no match,
    None will be returned.
    '''
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return neighborhoods['borough'][idx]
    return None
    
    
def findDropoffZone(p, index, zones):
    '''
    findZone returned the ID of the shape (stored in 'zones' with
    'index') that contains the given point 'p'. If there's no match,
    None will be returned.
    '''
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return neighborhoods['neighborhood'][idx]
    return None
    
    
def processTrips(pid, records):
    '''
    Our aggregation function that iterates through records in each
    partition, checking whether we could find a zone that contain
    the pickup location.
    '''
    import csv
    import pyproj
    import shapely.geometry as geom
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index, zones = createIndex('neighborhoods.geojson')    
    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    counts = {}
    
    for row in reader:
           
        pickup_point = geom.Point(proj(float(row[5]), float(row[6])))
        dropoff_point = geom.Point(proj(float(row[9]), float(row[10])))

        # Look up a matching zone, and update the count accordly if
        # such a match is found
        pickup_zone = findPickupZone(pickup_point, index, zones)
        dropoff_zone = findDropoffZone(dropoff_point, index, zones)

        if (pickup_zone,dropoff_zone):
            counts[(pickup_zone,dropoff_zone)] = counts.get((pickup_zone,dropoff_zone), 0) + 1
       
    return counts.items()

if __name__=='__main__':
    sc = SparkContext()
    sc.textFile(sys.argv[2]) \
        .mapPartitionsWithIndex(processTrips) \
        .reduceByKey(lambda x,y: x+y) \
        .filter(lambda x : x[0][0] != None) \
        .filter(lambda x: x[0][1] != None) \
        .map(lambda x: (x[0][0], ((x[0][1]), x[1]))) \
        .sortBy(lambda x: x[1][1], ascending=False) \
        .groupByKey().mapValues(list) \
        .map(lambda x: (x[0],x[1][0:3])) \
        .sortByKey() \
        .mapPartitionsWithIndex(toCSV) \
        .saveAsTextFile(sys.argv[3])