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

from encointer.classes import Ceremony, City, World
import pytest

def test_ceremony_instantiate():
    assert Ceremony(None)

@pytest.fixture
def world():
    world = World()
    city = City(1, 2, 1)
    city.newcomers(9)
    world.cities.add(city)
    world.label_people()
    return world


def test_ceremony_single(world):
    cer = Ceremony(None)
    city = list(world.cities)[0]
    [cer.register(p) for p in city.citizens]
    cer.perform(world)


def test_ceremony_two(world):
    cer1 = Ceremony(None)
    city = list(world.cities)[0]
    [cer1.register(p) for p in city.citizens]
    cer1.perform(world)

    cer2 = Ceremony(cer1)
    [cer2.register(p) for p in city.citizens]
    cer2.perform(world)

