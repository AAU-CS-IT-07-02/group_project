from pyproj import Proj
from uppaalcontroller import UPPAALData

def UTMtoLatLong(utm, E, N):
    p = Proj(proj='utm',zone=utm, ellps='WGS84')
    lon, lat = p(E,N,inverse=True)
    return lat, lon

def LatLongtoUTM(lat, lon,utm=32):
    p = Proj(proj='utm', zone=utm, ellps='WGS84')
    E, N = p(lon,lat)
    return E, N

class Point:
    def __init__ (self, x, y):
        self.x = x
        self.y = y

class Pixel:

    def __init__(self, E, N, size=50):
        self.E = E
        self.N = N
        self.size = size

        self.capacityMap = {}

class Cell:

    def __init__(self, id, name, freq, dir, active, traffic):
        self.id = id
        self.name = name
        self.freq = Cell.freqCharToNum(freq)
        self.power = Cell.freqToPower(self.freq)
        self.dir = dir
        self.active = active
        self.traffic = traffic
        self.trafficMap = {}
        self.trafficIndex = 0

    @staticmethod
    def freqCharToNum(c):
        return {'E' : 800,
                'V' : 900,
                'T' : 1800,
                'A' : 2100,
                'L' : 2600}.get(c)
    @staticmethod
    def freqToPower(freq):
        return {800 : 0.8,
            900 : 0.9,
            1800 : 1.8,
            2100 : 2.1,
            2600 : 2.6}.get(freq)

    def distributeTrafficUniform(self, index):
        # Traffic len 0 should never be the case
        # Trafficmap keys map be 0 if cell does not influence any pixel
        if len(self.traffic) == 0 or len(self.trafficMap.keys()) == 0:
            raise ValueError("Traffic 0 or trafficmap 0 for a cell")

        t = self.traffic[index] / len(self.trafficMap.keys())
        for k in self.trafficMap.keys():
            self.trafficMap[k] = t

class BaseStation:

    def __init__(self, id, name, latitude, longitude):
        self.id = id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.cells = []

class World:

    def __init__(self, lowerLeft, upperRight, duration):
        self.lowerLeft = lowerLeft
        self.upperRight = upperRight
        self.pixels = []
        self.stations = []
        self.duration = duration
        self.time = 0
        self.activeCells = set()
        self.cellCount = 0


    def restart(self):
        self.time = 0

    def applyStrat(self, strat):
        for b in self.stations:
            for c in b.cells:
                if c.id in strat:
                    now_active = strat[c.id]
                    if c.active and not now_active:
                        self.activeCells.remove(c.id)
                        c.active = False
                    elif not c.active and now_active:
                        self.activeCells.add(c.id)
                        c.active = True

    def evolve(self, seconds):
        pastHours = int(self.time / 3600)
        self.time += seconds
        presentHour = int(self.time / 3600)

        if pastHours != presentHour:
            print("Time is now " + str(self.time) + ", updating traffix index to " + str(presentHour))
            for b in self.stations:
                for c in b.cells:
                    c.distributeTrafficUniform(presentHour) 
            return True
        
        return False

    def getTotalPixelTraffic(self):
        t = {}

        for b in self.stations:
            for c in b.cells:
                for (E, N) in c.trafficMap:
                    if (E,N) in t:
                        t[(E,N)] += c.trafficMap[(E,N)]
                    else:
                        t[(E,N)] = c.trafficMap[(E,N)]
        return t

    def getMaxPixelCapacity(self, p):
        activeCapacity = [p.capacityMap[k] for k in p.capacityMap if k in self.activeCells]

        if len(activeCapacity) > 0:
            return max(activeCapacity)
        else:
            return 0
