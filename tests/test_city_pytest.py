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


from encointer.classes import City, Location, Person


def test_city_instantiate():
    assert City(1, 1, 1)


def test_generate_newcomer():
    city = City(1, 2, 3)
    assert City.location
    assert len(city.citizens) == 0
    city.newcomers(1)
    assert len(city.citizens) == 1
    city.newcomers(3)
    assert len(city.citizens) == 4



