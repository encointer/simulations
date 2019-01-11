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

from encointer.classes import Meetup, Person
from encointer.constants import *


def test_meetup_mustfail_too_few_people():
    p = Person(1, 2, 3)
    try:
        Meetup(set([p]))
    except:
        return
    raise NameError('a meetup with too few participants should  fail')


def test_meetup_mustfail_too_many_people():
    group = set()
    for k in range(MAX_MEETUP+1):
        group.add(Person(1,2,3))
    try:
        Meetup(group)
    except:
        return
    raise NameError('a meetup with too many participants (%i) should fail' % len(group))


def test_meetup_mustfail_all_have_met_before():
    group1 = set()
    for k in range(MAX_MEETUP):
        group1.add(Person(1,2,3))
    m = Meetup(group1)
    for p in group1:
        p.meetup_history.append(m)
    try:
        Meetup(group1)
    except:
        return
    raise NameError('a meetup with the same participants as last time should fail')


def test_meetup_few_have_met_before():
    group = set()
    for k in range(MIN_MEETUP):
        p = Person(1, 2, 3)
        p.label = str(k)
        group.add(p)
    m = Meetup(group)
    for p in group:
        p.meetup_history.append(m)
    # drop two of three
    group.pop()
    group.pop()
    # add two more people
    for k in range(2):
        group.add(Person(1, 2, 3))
    Meetup(group)
    # TODO: seems to wrongly count pairings

