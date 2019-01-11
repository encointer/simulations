#  Copyright (c) 2019 Alain Brenzikofer
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


from encointer.constants import *
import numpy as np
import networkx as nx
import shapely.geometry as sg


class Location:
    def __init__(self, x, y):
        self.coord = np.array([x,y])
    def distance_to(self, loc):
        return np.linalg.norm(loc.coord - self.coord)
    @classmethod
    def random_around_location(cls, min, max):
        assert isinstance(cls, Location)
        _rx = np.random.normal(cls.coord[0], max-min)
        _ry = np.random.normal(cls.coord[1], max-min)
        return Location(_x, _y, (np.random.random() / 2 + .5) * self.r * 2)


class Locatable:
    def __init__(self):
        self.location = None

    @property
    def location(self):
        return self.__location

    @location.setter
    def location(self, loctuple):
        assert len(loctuple) == 2
        self.__location = loctuple

    def lat(self):
        return self.location[1]

    def lon(self):
        return self.location[0]

    def distance_to(self, other):
        assert isinstance(other, Locatable)
        return np.linalg.norm(np.array(other.location) - np.array(self.location))

    def move_to(selfself, other):
        pass  # TODO


class Person(Locatable):
    def __init__(self, x, y, r):
        self.location = (x,y)
        self.r = r
        self.label = None
        self.seed_history = list()
        self.seed_preference = list() #which currency color should be minted by preference?
        self.meetup_history = list()
        #self.neighbors = set()

    def last_meetup_buddies(self):
        if self.meetup_history:
            return self.meetup_history[-1].participants.difference(set([self]))
        else:
            return set()

    def has_met(self, p):
        return p in self.last_meetup_buddies()


class City(Locatable):
    def __init__(self, x, y, r):
        self.location = (x, y)
        self.r = r
        self.citizens = set()
        self.seed = None


    def newcomers(self, n):
        for iter in range(n):
            _x = np.random.normal(self.lon(), self.r)
            _y = np.random.normal(self.lat(), self.r)
            _p = Person(_x, _y, (np.random.random() / 2 + .5) * self.r * 2)
            self.citizens.add(_p)


class World:
    def __init__(self):
        self.cities = set()
        self.coins = Coins()

    def label_people(self):
        index=1
        for c in self.cities:
            for p in c.citizens:
                p.label = str(index)
                index += 1


class Ceremony:
    def __init__(self, parent):
        self.parent = parent
        self.registry = set()
        self.orphans = set()
        self.meetups = []

    def register(self, person):
        assert isinstance(person, Person)
        self.registry.add(person)


    def assign_meetups(self):
        ## create an undirected graph with all possible matchings subject to range limits
        G = nx.Graph()
        for p in self.registry:
            G.add_node(p)
        print("registered are")
        print([p.label for p in self.registry])
        for p in self.registry:
            assert isinstance(p, Person)
            for q in self.registry:
                if p.distance_to(q) < p.r + q.r:
                    if (p != q):
                        G.add_edge(p, q)
        Gcpy = G.copy()
        groups = []
        while True:
            ## start assigning nodes of low degree
            ndeg = G.degree(G.nodes)
            if not ndeg:
                break
            # degree <2 has no chance to meet
            ordered_nodes = []
            for deg in range(2, 1 + max([value for key, value in ndeg])):
                _ps = [key for key, value in ndeg if value == deg]
                # if there are multiple, sort by number of connected buddies
                # who has many former buddies now is more difficult to assign
                _psb = [(p, len(p.last_meetup_buddies())) for p in _ps]
                _psb.sort(key=lambda x: x[1], reverse=True)
                ordered_nodes = ordered_nodes + [p for (p, n) in _psb]
            if not ordered_nodes:
                break
            print("assignement order:")
            print([p.label for p in ordered_nodes])
            # find all cliques per node and group
            # low degrees first
            _p = ordered_nodes[0]
            print("assigning %s" % _p.label)
            _candidates = []
            for clq in nx.cliques_containing_node(G, _p):
                print("evaluating clique")
                print([p.label for p in clq])
                if len(clq) < 3:
                    continue
                clq.remove(_p)
                _psdeg = [(p, _p.has_met(p), ndeg[p]) for p in clq]
                _psdeg.sort(key=lambda x: (x[1], x[2]), reverse=False)
                print([(d[0].label, d[1], d[2]) for d in _psdeg])
                if ((_psdeg[0][1] == True) or (_psdeg[1][1] == True and _psdeg[0][0].has_met(_psdeg[1][0]))):
                    print("too many have met before. dropping")
                    continue
                _candidates.append((set([_p, _psdeg[0][0], _psdeg[1][0]]), _psdeg[0][1] + _psdeg[1][1]))
                print("found candidate with malus %i" % _candidates[-1][1])
                print([p.label for p in _candidates[-1][0]])
            if _candidates:
                _candidates.sort(key=lambda x: x[1])
                groups.append(_candidates[0][0])
                # now remove these from graph before iterating
                G.remove_nodes_from(groups[-1])

            else:
                # no meetups possible for this person
                print("cannot assign %s" % _p.label)
                G.remove_node(_p)

        _orphans = self.registry.difference(set([p for l in groups for p in l]))
        # now lets assign orphans if possible
        for _p in _orphans:
            for m in groups:
                try:
                    Meetup(m.union(set([_p]))) # fails if invalid
                    m.add(_p)
                    print("found a meetup for orphan %s" % _p.label)
                    break
                except:
                    pass

        # refresh
        _orphans = self.registry.difference(set([p for l in groups for p in l]))

        print("%i meetups:" % len(groups))
        for m in groups:
            print([p.label for p in m])
        print("%i orphans:" % len(_orphans))
        print([p.label for p in _orphans])

        # instantiate meetup objects and find adjacent meetups used for grouping the minting
        print("link adjacent meetups")
        _meetupobjs = [Meetup(m) for m in groups]
        for mo in _meetupobjs:
            mo.adjacent_meetups = set()
            for mu in _meetupobjs:
                if mo != mu and nx.has_path(Gcpy, list(mo.participants)[0], list(mu.participants)[0]):
                    mo.adjacent_meetups.add(mu)
        self.meetups = _meetupobjs
        self.orphans = _orphans

    def assign_seeds(self, world):
        # create graph to group islands
        G = nx.Graph()
        G.add_nodes_from(self.meetups)
        for m in self.meetups:
            for n in m.adjacent_meetups:
                G.add_edge(m, n)
        islands = [set(g.nodes) for g in nx.connected_component_subgraphs(G)]
        for island in islands:
            people = [p for m in island for p in m.participants]
            # see if there is a seed already
            seedprefs = [p.seed_preference[0] for p in people if p.seed_preference]

            # find closest city
            m = list(island)[0]
            cit = [(c, m.distance_to(c)) for c in world.cities]
            cit.sort(key=lambda x: x[1])
            icity = cit[0][0]
            if not seedprefs:
                # assign new seed to everybody
                seed = Seed(self, m.location)
                world.coins.balances[seed] = dict()
                icity.seed = seed  # might be overwritten by other island. don't care for now
                for p in people:
                    p.seed_history.append(seed)
                    p.seed_preference.append(seed)  # we assume loyalty with home town by default
                for m in island:
                    m.seeds = [seed]
            else:
                for m in island:
                    # collect seeds linked directly or indirectly to this meetup
                    m.seeds = [s for p in m.participants for s in p.seed_history]
                    for p in people:
                        if not p.seed_history:
                            # adopt the seed preference from most other participants
                            l = [p.seed_preference[0] for p in m.participants if p.seed_preference]
                            seed = max([(s, l.count(s)) for s in l])[0]
                            p.seed_history.append(seed)
                            p.seed_preference.append(seed)

    def perform(self, world):
        self.assign_meetups()
        self.assign_seeds(world)
        for m in self.meetups:
            assert isinstance(m, Meetup)
            for p in m.participants:
                assert isinstance(p, Person)
                p.meetup_history.append(m)
                # only allow seeds that were visited at least
                # once before by one meetup participant
                seeds = [s for s in p.seed_preference if s in m.seeds]
                if not seeds:
                    print("dismissed %i from minting" % p.id)
                    print(m.seeds)
                    print([s.location for s in m.seeds])
                    print(p.seed_preference[0].location)
                    continue
                seed = seeds[0]
                # minting for the seed of this meetup
                world.coins.mint(seed, p)


class Meetup(Locatable):
    def __init__(self, participants):
        self.participants = participants
        self.adjacent_meetups = set()
        self.seeds = set()
        assert self.is_valid()
        self.decide_location()

    def is_valid(self):
        m = self.participants
        print("checking validity of meetup")
        print([p.label for p in m])
        if len(m) < MIN_MEETUP:
            return False
        if len(m) > MAX_MEETUP:
            return False
        havemet = 0
        for _p in m:
            assert isinstance(_p, Person)
            for _q in m.difference(set([_p])):
                assert isinstance(_q, Person)
                havemet = havemet + _p.has_met(_q)
                if _p.distance_to(_q) > _p.r+_q.r:
                    return False
        print("out of %i participants, %i pairings have met at the last ceremony" % (len(m), havemet/2))
        # havemet counts each encounter twice!
        if havemet/2 > len(m)*MEETUP_REVISIT_RATIO:
            return False
        return True

    def decide_location(self):
        circles = [sg.Point(p.lon(), p.lat()).buffer(p.r * RANGE_TOLERANCE) for p in self.participants]
        target = circles.pop()
        for c in circles:
            target = target.intersection(c)
        self.location = target.centroid.coords[0]

class Seed(Locatable):
    def __init__(self, ceremony, location):
        self.ceremony = ceremony
        self.location = location


class Coins:
    def __init__(self):
        self.balances = dict()

    def get_balances(self, person):
        bal = dict()
        for s in self.balances.keys():
            if person in self.balances[s]:
                bal[s] = self.balances[s][person]
        return bal

    def mint(self, seed, person):
        if seed not in self.balances.keys():
            self.balances[seed] = dict()
            self.balances[seed][person] = REWARD
            return
        if person in self.balances[seed].keys():
            self.balances[seed][person] += REWARD
        else:
            self.balances[seed][person] = REWARD

