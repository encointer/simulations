#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alain Brenzikofer, encointer.org

This simulation creates populations of persons in cities, assigns people 
to meetups according to encointer rules for successuve ceremonies and 
mints coins.
Along the way it plots explanatory figures for meetup assignments

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import shapely.geometry as sg
import matplotlib.pyplot as plt
import descartes
import networkx as nx

MIN_MEETUP = 3 
MAX_MEETUP = 12
# share of participants that may have met at the last ceremony
MEETUP_REVISIT_RATIO=1.0/3.0

# meetup range intersection is only checked pairwise, not real intersection
RANGE_TOLERANCE = 1.2

REWARD = 1.0

# all people allover the world that have participated once or more times
population = set()

# all people registered for the next ceremony
registry = set()
meetups = []

cities = set()

#dict[Seed][Person]
wallets = dict()

class Seed:
    def __init__(self, ceremony, location):
        self.ceremony = ceremony
        self.location = location
        self.successor=None
        
class Ceremony:
    def __init__(self, last, meetups):
        self.meetups=meetups
        self.last = last
        if not last:
            self.id = 1
        else:
            self.id = last.id+1
        self.assign_seeds()
            
    def assign_seeds(self):
        #create graph to group islands
        G = nx.Graph()
        G.add_nodes_from(self.meetups)
        for m in self.meetups:
            for n in m.adjacent_meetups:
                G.add_edge(m,n)
        islands = [set(g.nodes) for g in nx.connected_component_subgraphs(G)]
        for island in islands:
            people = [p for m in island for p in m.participants]
            #see if there is a seed already
            seedprefs = [p.seed_preference[0] for p in people if p.seed_preference]

            #find closest city
            iloc = list(island)[0].location
            cit = [(c, np.linalg.norm(iloc-c.pos)) for c in cities]
            cit.sort(key = lambda x: x[1])
            icity = cit[0][0]
            if not seedprefs:
                #assign new seed to everybody
                seed = Seed(self, iloc)
                wallets[seed] = dict()
                icity.seed = seed # might be overwritten by other island. don't care for now
                for p in people:
                    p.seeds.append(seed)
                    p.seed_preference.append(seed) # we assume loyalty with home town by default
                for m in island:
                    m.seeds = [seed]
            else:
                for m in island:
                    #collect seeds linked directly or indirectly to this meetup
                    m.seeds = [s for p in m.participants for s in p.seeds]
                    for p in people:
                        if not p.seeds:
                            # adopt the seed preference from most other participants
                            l=[p.seed_preference[0] for p in m.participants if p.seed_preference]
                            seed = max([(s, l.count(s)) for s in l])[0]
                            p.seeds.append(seed)
                            p.seed_preference.append(seed)
                            
        
        
class Meetup:
    def __init__(self, m):
        for p in m:
            assert(isinstance(p, Person))
        self.participants = m
        self.adjacent_meetups = set()
        self.location = ()
        self.seeds = set()
       
    
class Person:
    def __init__(self, x, y, r):
        self.pos = np.array([x,y])
        self.r=r
        self.id = len(population)+1
        #self.wallet = dict() #person maintains one balance per seed in a dict
        self.seeds = list()  
        self.seed_preference = list() #which currency color should be minted by preference?
        self.meetup_buddies = list()
        self.neighbors = set()
        
    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]
        
    def has_met(self, p):
        return (p in self.meetup_buddies)
    def balances(self):
        return [(s.location, wallets[s][self]) if (self in wallets[s]) else (s.location, 0.0) for s in wallets.keys() ]
            
        
class City:
    
    def __init__(self, x, y, r):
        self.pos=np.array([x,y])
        self.r=r
        self.citizens = set()
        self.seed = None

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]
        
    def newcomers(self, n):
        for iter in range(n):
            _x = np.random.normal(self.x, self.r)
            _y = np.random.normal(self.y, self.r)
            _p = Person(_x,_y,(np.random.random()/2+.5)*self.r*2)
            population.add(_p)
            self.citizens.add(_p)
## plotting functions
def plot_population():
    plt.clf()
    ax = plt.gca()
    plt.hold(True)
    for p in population:
        assert isinstance(p,Person)
        pnt = sg.Point(p.x,p.y).buffer(p.r*RANGE_TOLERANCE)
        ax.add_patch(descartes.PolygonPatch(pnt, fc='b', ec='k', alpha=0.2))
        plt.text(p.x,p.y,p.id, ha='center', va='center')
    
    ax.set_aspect('equal')
    ax.set_xlim(-5, 10); ax.set_ylim(-5, 5)
    plt.title('population')
    plt.show()
    plt.savefig("sim-communities-fig" + str(plt.gcf().number)+ ".svg")

def print_wallets():
    for seed in wallets.keys():
        print("Balances for seed at %f,%f" % seed.location)
        for p, bal in wallets[seed].items():
            print("id %i: %f" % (p.id, bal))


def decide_meetup_locations(meetups):
    cmap = list(plt.get_cmap('Set1').colors)[::-1]
    
    for m in meetups:
        plt.figure()
        ax = plt.gca()
        try:
            col= cmap.pop()
        except:
            pass
        circles = [sg.Point(p.x,p.y).buffer(p.r*RANGE_TOLERANCE) for p in m.participants]
        for c in circles:
            ax.add_patch(descartes.PolygonPatch(c, fc=col, alpha=0.1))
        target = circles.pop()
        for c in circles:
            target=target.intersection(c)
        print("intersecting meetup")
        print([p.id for p in m.participants])
        ax.add_patch(descartes.PolygonPatch(target, fc=col, ec='k', alpha=0.6))
        ax.hold(True)
        for p in m.participants:
            ax.plot(p.x,p.y, color='white', marker='o', markersize=10)
            plt.text(p.x,p.y,p.id, ha='center', va='center')
            for q in m.participants:
                if q in p.meetup_buddies:
                    ax.add_line(plt.Line2D((p.x, q.x), (p.y, q.y)))
        loc = target.centroid.coords[0]
        ax.plot(loc[0],loc[1], color='k', marker='+')
        plt.title('ceremony %i meetup' % ceremony_id)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 10); ax.set_ylim(-5, 5)
        plt.xlabel('longitude')    
        plt.ylabel('latitude')    
        plt.show()
        plt.savefig("sim-communities-fig" + str(plt.gcf().number)+ ".svg")
        m.location = loc

    

def plot_orphans(orphans):
    plot_population()
    plt.hold(True)
    for p in orphans:
        plt.plot(p.x,p.y,'rx')
    plt.title("Number of orphans: %i" % len(orphans))    
    print("number of orphans: %i" % len(orphans))

def takeSecond(elem):
    return elem[1]

def is_valid_meetup(m):
    print("checking validity of meetup")
    print([p.id for p in m])
    if len(m)<MIN_MEETUP:
        return False
    if len(m)>MAX_MEETUP:
        return False
    havemet=0
    for _p in m:
        for _q in m.difference(set([_p])):
            havemet = havemet + _p.has_met(_q)
            if np.linalg.norm(_p.pos-_q.pos) > _p.r+_q.r:
                return False
    print("out of %i participants, %i pairings have met at the last ceremony" % (len(m), havemet/2))
    #havemet counts each encounter twice!
    if havemet/2 > len(m)*MEETUP_REVISIT_RATIO:
        return False
    return True
            
## graph partitioning
def assign_meetups():
    ## create an undirected graph with all possible matchings subject to range limits

    G = nx.Graph()
    for p in registry:
        G.add_node(p)
    print("registered are")
    print([p.id for p in registry])
    for p in registry:
        for q in registry:
            if np.linalg.norm(p.pos-q.pos) < p.r+q.r:
                if (p != q):
                    G.add_edge(p,q)
    Gcpy = G.copy()
    _meetups = []
    while True:
        ## start assigning nodes of low degree
        ndeg = G.degree(G.nodes)
        if not ndeg:
            break
        #degree <2 has no chance to meet
        ordered_nodes  = []
        for deg in range(2,1+max([value for key,value in ndeg])):
            _ps = [key for key, value in ndeg if value == deg]
            # if there are multiple, sort by number of connected buddies
            # who has many former buddies now is more difficult to assign
            _psb = [(p, len(p.meetup_buddies)) for p in _ps]
            _psb.sort(key=takeSecond, reverse=True)
            ordered_nodes = ordered_nodes + [p for (p,n) in _psb]
        if not ordered_nodes:
            break
        print("assignement order:")
        print([p.id for p in ordered_nodes])
        # find all cliques per node and group
        # low degrees first
        _p = ordered_nodes[0]
        print("assigning %i" % _p.id)
        _candidates=[]
        for clq in nx.cliques_containing_node(G,_p):
            print("evaluating clique")
            print([p.id for p in clq])
            if len(clq)<3:
                continue
            clq.remove(_p)
            _psdeg = [(p, _p.has_met(p), ndeg[p]) for p in clq]
            _psdeg.sort(key=lambda x: (x[1], x[2]), reverse=False)
            print([(d[0].id, d[1], d[2]) for d in _psdeg])
            if ((_psdeg[0][1]==True) or (_psdeg[1][1]==True and _psdeg[0][0].has_met(_psdeg[1][0]))):
                print("too many have met before. dropping")
                continue
            _candidates.append((set([_p, _psdeg[0][0], _psdeg[1][0]]),_psdeg[0][1]+_psdeg[1][1]))
            print("found candidate with malus %i" % _candidates[-1][1])
            print([p.id for p in _candidates[-1][0]])
        if _candidates:
            _candidates.sort(key=lambda x: x[1])
            _meetups.append(_candidates[0][0])
            # now remove these from graph before iterating
            G.remove_nodes_from(_meetups[-1])
            
        else:
            #no meetups possible for this person
            print("cannot assign %i" % _p.id)
            G.remove_node(_p)
        
 
    _orphans = registry.difference(set([p for l in _meetups for p in l])) 
    # now lets assign orphans if possible
    for _p in _orphans:
        for m in _meetups:
            if is_valid_meetup(m.union(set([_p]))):
                m.add(_p)
                print("found a meetup for orphan %i" % _p.id)
                break
    #refresh
    _orphans = registry.difference(set([p for l in _meetups for p in l])) 
    for m in _meetups:
        assert(is_valid_meetup(m))
    print("%i meetups:" % len(_meetups))                         
    for m in _meetups:
        print([p.id for p in m])
    print("%i orphans:" % len(_orphans))
    print([p.id for p in _orphans])
    
    #instantiate meetup objects and find adjacent meetups used for grouping the minting
    print("link adjacent meetups")
    _meetupobjs = [Meetup(m) for m in _meetups]
    for mo in _meetupobjs:
        mo.adjacent_meetups = set()
        for mu in _meetupobjs:
            if mo != mu and nx.has_path(Gcpy, list(mo.participants)[0], list(mu.participants)[0]):
                mo.adjacent_meetups.add(mu)
    return (_meetupobjs, _orphans)
    
def perform_ceremony(ceremony):
    for p in population:
        p.meetup_buddies = set()
    for m in ceremony.meetups:
        for p in m.participants:
            buddies= m.participants.copy()
            buddies.remove(p)
            p.meetup_buddies = buddies
            
            #only allow seeds that were visited at least 
            #once before by one meetup participant
            seeds = [s for s in p.seed_preference if s in m.seeds]
            if not seeds:
                print("dismissed %i from minting" % p.id)
                print(m.seeds)
                print([s.location for s in m.seeds])
                print(p.seed_preference[0].location)
                continue
            seed=seeds[0]
            #minting for the seed of this meetup
            if p in wallets[seed].keys():
                wallets[seed][p] += REWARD
            else:
                wallets[seed][p] = REWARD
                
city1 = City(0,0,1)        
cities.add(city1)
city1.newcomers(9)


plt.close('all')
plt.figure()
plot_population()

#everybody regsiters
registry = population.copy()
print("%%%%%%% new ceremony%%%%%%%%%")
ceremony_id=1
(meetups,orphans)=assign_meetups()
decide_meetup_locations(meetups)
cer = Ceremony(None,meetups)
plt.figure()
plot_orphans(orphans)
perform_ceremony(cer)

#new participants
city2 = City(10,0,1)
cities.add(city2)
city2.newcomers(9)

print("%%%%%%% new ceremony%%%%%%%%%")
registry = population.copy()
ceremony_id = 2
(meetups,orphans)=assign_meetups()
decide_meetup_locations(meetups)
cer = Ceremony(cer,meetups)
plt.figure()
plot_orphans(orphans)
perform_ceremony(cer)

#one person emigrates to the other city
expat = list(city2.citizens)[0]
expat.pos = city1.pos
#expat adopts new local currency
expat.seed_preference = [city1.seed] + expat.seed_preference

#another person just visits the other city. not changing seed preference
visitor=list(city1.citizens)[0]
visitor.pos = city2.pos


print("%%%%%%% new ceremony%%%%%%%%%")
registry = population.copy()
ceremony_id = 3
(meetups,orphans)=assign_meetups()
decide_meetup_locations(meetups)
cer = Ceremony(cer,meetups)
plt.figure()
plot_orphans(orphans)
perform_ceremony(cer)

print_wallets()