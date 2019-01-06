#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:00:42 2019

@author: brenzi
"""
import numpy as np
import shapely.geometry as sg
import matplotlib.pyplot as plt
import descartes
import networkx as nx

MIN_MEETUP = 3 
OPT_MEETUP = 8
MAX_MEETUP = 12
# share of participants that may have met at the last ceremony
MEETUP_REVISIT_RATIO=0.333

# all people allover the world that have participated once or more times
population = set()

# all people registered for the next ceremony
registry = set()
meetups = []

cities = set()


#class ceremony:

#class meetup:
    
class Person:
    neighbors = set()
    meetup_buddies = list()
    
    def __init__(self, x, y, r):
        self.pos = np.array([x,y])
        self.r=r
        self.id = len(population)+1
        
    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]
        
    def has_met(self, p):
        return (p in self.meetup_buddies)
        
class City:
    citizens = set()
    def __init__(self, x, y, r):
        self.x=x
        self.y=y
        self.r=r
    def newcomers(self, n):
        for iter in range(n):
            _x = np.random.normal(self.x, self.r)
            _y = np.random.normal(self.y, self.r)
            _p = Person(_x,_y,(np.random.random()/2+.9)*self.r*2)
            population.add(_p)
            self.citizens.add(_p)
## plotting functions
def plot_population():
    plt.clf()
    ax = plt.gca()
    plt.hold(True)
    for p in population:
        assert isinstance(p,Person)
        pnt = sg.Point(p.x,p.y).buffer(p.r)
        ax.add_patch(descartes.PolygonPatch(pnt, fc='b', ec='k', alpha=0.2))
        plt.text(p.x,p.y,p.id, ha='center', va='center')
    
    ax.set_aspect('equal')
    ax.set_xlim(-5, 10); ax.set_ylim(-5, 5)
    plt.title('population')
    plt.show()
    plt.savefig("sim-communities-fig" + str(plt.gcf().number)+ ".svg")

def decide_meetup_locations(meetups):
    cmap = list(plt.get_cmap('Set1').colors)[::-1]
    meetup_locations = []
    for m in meetups:
        plt.figure()
        ax = plt.gca()
        try:
            col= cmap.pop()
        except:
            pass
        circles = [sg.Point(p.x,p.y).buffer(p.r) for p in m]
        for c in circles:
            ax.add_patch(descartes.PolygonPatch(c, fc=col, alpha=0.1))
        target = circles.pop()
        for c in circles:
            target=target.intersection(c)
        ax.add_patch(descartes.PolygonPatch(target, fc=col, ec='k', alpha=0.6))
        ax.hold(True)
        for p in m:
            ax.plot(p.x,p.y, color='white', marker='o', markersize=10)
            plt.text(p.x,p.y,p.id, ha='center', va='center')
            for q in m:
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
        meetup_locations.append(loc)
    print("%i Meetups" % len(meetups))
    return meetup_locations

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
    for _p in m:
        for _q in m.difference(set(_p)):
            havemet = havemet + _p.has_met(_q)
            if np.linalg.norm(_p.pos-_q.pos) > _p.r+_q.r:
                return False
    if havemet > len(m)*MEETUP_REVISIT_RATIO:
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
            if _psdeg[0][1]==True:
                print("all have met before. dropping")
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
            if is_valid_meetup(m.union(set(_p))):
                m.add(_p)
                print("found a meetup for orphan %i" % _p)
                break
    
    print("%i meetups:" % len(_meetups))                         
    for m in _meetups:
        print([p.id for p in m])
    print("%i orphans:" % len(_orphans))
    print([p.id for p in _orphans])
    return (_meetups, _orphans)
    
def perform_meetups(meetups):
    for p in population:
        p.meetup_buddies = set()
    for m in meetups:
        for p in m:
            buddies= m.copy()
            buddies.remove(p)
            p.meetup_buddies = buddies
                
city1 = City(0,0,1)        
cities.add(city1)
city1.newcomers(9)

#city2.newcomers(3)
#city2 = City(5,0,1)
#cities.add(city2)

plt.close('all')
plt.figure()
plot_population()

#everybody regsiters
registry = population.copy()
ceremony_id = 1
(meetups,orphans)=assign_meetups()
meetup_locations= decide_meetup_locations(meetups)
plt.figure()
plot_orphans(orphans)

perform_meetups(meetups)

# second ceremony
print("%%%%%%% new ceremony%%%%%%%%%")
registry = population.copy()
ceremony_id = 2
(meetups,orphans)=assign_meetups()
meetup_locations= decide_meetup_locations(meetups)
plt.figure()
plot_orphans(orphans)
perform_meetups(meetups)
# third ceremony
print("%%%%%%% new ceremony%%%%%%%%%")
registry = population.copy()
ceremony_id = 3
(meetups,orphans)=assign_meetups()
meetup_locations= decide_meetup_locations(meetups)
plt.figure()
plot_orphans(orphans)
perform_meetups(meetups)