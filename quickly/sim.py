from world import Point, World, Cell
from dummycontrollers import RandomController

class Simulator:

    def __init__(self, world, stepSize = 60,      # stepSize must be less than world.duration to make sense
                 controlPoint = 60*60, penalty = 0.5, # penalty per time unit
                 controller = RandomController()):
        self.world = world
        self.stepSize = stepSize
        self.horizon = world.duration
        self.controlPoint = controlPoint
        self.penalty = penalty
        self.controller =  controller

    def restart(self):
        self.world.restart()

    def run(self):
        energy,pen,rew = self.computeReward()
        
        accum_rew = 0
        accum_pen = 0
        accum_energy = 0
        control_update = False
        traffic_update = False

        for i in range(self.stepSize, self.world.duration,self.stepSize):

            if traffic_update or control_update:
                print("Computing reward at time: {}".format(i))
                energy,pen,rew = self.computeReward()
                control_update = False
                traffic_update = False

            if i > 0 and i % self.controlPoint == 0:
                print("Calling Controller")
                strat = self.controller.getActions(self.world)
                print("Computed strategy: " + str(strat))
                self.world.applyStrat(strat)
                control_update = True

            accum_rew += rew
            traffic_update = self.doStep()

            
        accum_rew += rew # reward from last self.doStep()
        accum_pen += pen
        accum_energy += energy
        print("End Reward:{}".format(accum_rew))
        print("End Energy:{}".format(accum_energy))
        print("End Penalty:{}".format(accum_pen))
        return accum_energy, accum_pen, accum_rew
         
    def doStep(self):
        return self.world.evolve(self.stepSize)


    def computeReward(self):
        penalty = 0
        energy = 0

        pixelTraffic = self.world.getTotalPixelTraffic()

        for p in self.world.pixels:
            t = pixelTraffic[(p.E,p.N)]
            c = self.world.getMaxPixelCapacity(p)


            if c < t:
                penalty += self.penalty
            
        for b in self.world.stations:
            for c in b.cells:
                if c.active:
                    energy += Cell.freqToPower(c.freq)
        print("Energy:{}".format(energy))
        print("Penalty:{}".format(penalty))
        return energy*self.stepSize, penalty*self.stepSize, (penalty + energy) * self.stepSize
