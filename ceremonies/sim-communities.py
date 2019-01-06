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
        pnt = sg.Point(p.x,p.y).buffer(p.r)
        ax.add_patch(descartes.PolygonPatch(pnt, fc='b', ec='k', alpha=0.2))
        plt.text(p.x,p.y,p.id, ha='center', va='center')
    
    ax.set_aspect('equal')
    ax.set_xlim(-5, 10); ax.set_ylim(-5, 5)
    plt.title('population')
    plt.show()

def decide_meetup_locations(meetups):
    cmap = list(plt.get_cmap('Set1').colors)[::-1]
    meetup_locations = []
    for m in meetups:
        plt.figure()
        ax = plt.gca()

        col= cmap.pop()
        circles = [sg.Point(p.x,p.y).buffer(p.r) for p in m]
        for c in circles:
            ax.add_patch(descartes.PolygonPatch(c, fc=col, alpha=0.1))
        target = circles.pop()
        for c in circles:
            target=target.intersection(c)
        ax.add_patch(descartes.PolygonPatch(target, fc=col, ec='k', alpha=0.6))
        ax.hold(True)
        for p in m:
            ax.plot(p.x,p.y, color=col, marker='d')
            plt.text(p.x,p.y,p.id, ha='center', va='center')
            for q in m:
                if q in p.meetup_buddies:
                    ax.add_line(plt.Line2D((p.x, q.x), (p.y, q.y)))
        loc = target.centroid.coords[0]
        ax.plot(loc[0],loc[1], color='k', marker='+')
        plt.title('ceremony %i meetup' % ceremony_id)
        plt.show()
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
    
    ## graph partitioning into fully connected subgraphs
    _cliques = list(nx.find_cliques(G))
    # take maximal clique
    _clique_subgraphs = [max(_cliques, key=len)]
    print("maximal clique: %%%%%%%%%%%%%%%%%%")
    print([p.id for p in _clique_subgraphs[0]])

    for c in _cliques:
        print("checking clique")
        print([p.id for p in c])
        #if c shares no persons with booked cliques
        if not set([p for l in _clique_subgraphs for p in l])&set(c):
            _clique_subgraphs.append(c)
            print("added")

    _meetups = []
    _orphans = registry.difference(set([p for l in _clique_subgraphs for p in l])) 
    for c in _clique_subgraphs:
        print("found disjoint clique")
        print([p.id for p in c])

        if len(c)<MIN_MEETUP:
            _orphans = _orphans.union(c)
            continue
        #get unfrozen copy of subgraph
        Gc = nx.Graph(G.subgraph(c))
        # now find all pairs that have met at the last ceremony and cut the graph there
        for p in c:
            assert(isinstance(p, Person))
            for buddy in p.meetup_buddies:
                try:
                    Gc.remove_edge(p,buddy)
                except:
                    pass
                    
        lc = list(nx.find_cliques(Gc))
        # how many persons are in a clique < MIN_MEETUP
        #n_orphans = sum([n for n in [len(l) for l in lc] if n < MIN_MEETUP])
        clique_orphans = set()
        clique_meetups=[]
        # first, create smallest possible meetups with people that haven't met in the last round
        for cc in lc:
            n = len(cc)
            if n< MIN_MEETUP:
                for p in cc:
                    clique_orphans.add(p)
                continue
            while n >= MIN_MEETUP:
                clique_meetups.append(set())
                for i in range(MIN_MEETUP) if n>=2*MIN_MEETUP else range(n):
                    clique_meetups[-1].add(cc.pop())
                    n = n-1
        #now add one orphan to each meetup until limit
        #FIXME: this is very conservative, leaving unnecessary orphans
        for m in clique_meetups:
            for i in range(np.int_(np.floor(len(m)*MEETUP_REVISIT_RATIO))):
                if len(clique_orphans)==0:
                    print("all orphans could be placed")
                    break
                m.add(clique_orphans.pop())
                
        _meetups = _meetups + clique_meetups
        _orphans = _orphans.union(clique_orphans)
    # give orphans an option to go increase their radius
    
    
    print("%i meetups:" % len(_meetups))                         
    for m in _meetups:
        print([p.id for p in m])
    print("%i orphans:" % len(_orphans))
    print([p.id for p in _orphans])
    return (_meetups, _orphans)
    
def perform_meetups(meetups):
    for m in meetups:
        for p in m:
            buddies= m.copy()
            buddies.remove(p)
            p.meetup_buddies = buddies
                
city1 = City(0,0,1)        
cities.add(city1)
city2 = City(5,0,1)
cities.add(city2)
city1.newcomers(9)
city2.newcomers(3)

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
registry = population.copy()
ceremony_id = 2
(meetups,orphans)=assign_meetups()
meetup_locations= decide_meetup_locations(meetups)
plt.figure()
plot_orphans(orphans)
