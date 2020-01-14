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
import matplotlib.pyplot as plt

t = np.arange(0,356*8)
# days betwwen cermonies
period = np.int(41)
# per ceremony ubi / reward
reward = 10000
# per annum
halving_period = period*7

templ = reward*np.exp(-t*np.log(2)/halving_period)


bal = templ.copy()

for c in np.int_(np.arange(1,np.int_(np.ceil(t[-1]/period)))):
   bal[c*period:-1] += templ[0:-1-c*period]

plt.figure()
 
plt.plot(t[:-2],bal[:-2])
plt.xlabel('days')
plt.ylabel('money supply [tokens]')
#plt.title('nominal effect of demurrage with halving period = ' + str(halving_period) + ' days')
plt.xticks(np.arange(0,t[-1], period*10))
plt.grid('on')
plt.savefig("ubi-vs-demurrage-balance-" + str(plt.gcf().number)+ ".svg")

