"""
Author: Jangwon Park
Date: June 18, 2024

Contains the simulation logic of the Erlang-R queue.
"""

import SimFunctions
import SimRNG 
import SimClasses
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.close('all')

def t_mean_confidence_interval(data,alpha):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = stats.t.ppf(1-alpha/2, n-1)*se
    return m, m-h, m+h

def simulate(params):
    server_num = params['N']
    # mean_tba = 1/params['lambda']
    mean_ttr = 1/params['delta']
    mean_st = 1/params['mu']
    ReadmissionProb = params['p']
    # p_h = params['p_h']
    # p_l = params['p_l']

    # hold_cost = params['h']
    # readmit_cost = params['r']
    # interv_cost = params['C']

    X = 0
    Y = 0

    # SET-UP FOR THE SIMULATION
    Queue = SimClasses.FIFOQueue()
    Wait = SimClasses.DTStat()
    Orbit = SimClasses.CTStat()
    Congestion = SimClasses.CTStat()
    Readmissions = SimClasses.DTStat()
    Interventions = SimClasses.DTStat()
    Server = SimClasses.Resource()
    Server.SetUnits(server_num)
    Calendar = SimClasses.EventCalendar()

    TheCTStats = []
    TheDTStats = []
    TheQueues = []
    TheResources = []

    TheDTStats.append(Wait)
    TheDTStats.append(Readmissions)
    TheDTStats.append(Interventions)
    TheCTStats.append(Orbit)
    TheCTStats.append(Congestion)
    TheQueues.append(Queue)
    TheResources.append(Server)

    RunLength = 120.0 # in minutes
    WarmUp = 0.0

    def NSPP(Stream, params):
        t = SimClasses.Clock + SimRNG.Expon(1/params['max_rate'], Stream) # possible arrival time
        
       
        while SimRNG.Uniform(0, 1, Stream) >= (params['arrival_rate_fn'](t) / params['max_rate']):
            t = t + SimRNG.Expon(1/params['max_rate'], Stream)
            if t >= 120:
                # Doesn't matter what we return. Simulation will end at 120.
                return t - SimClasses.Clock
        nspp = t - SimClasses.Clock # inter-arrival time
        return nspp
    
    def Arrival(X):
        SimFunctions.Schedule(Calendar,"Arrival",NSPP(1, params))
        Customer = SimClasses.Entity()
        Queue.Add(Customer)

        Congestion.Record(max(X-server_num, 0))

        if Server.Busy < server_num:
            Server.Seize(1)
            SimFunctions.Schedule(Calendar,"EndOfService",SimRNG.Expon(mean_st,2))
        
        return X + 1
        
    def EndOfService(X, Y):
        DepartingCustomer = Queue.Remove()
        Wait.Record(SimClasses.Clock - DepartingCustomer.CreateTime)
        Congestion.Record(max(X-server_num, 0))

        # if there are customers waiting
        if Queue.NumQueue() >= server_num:
            SimFunctions.Schedule(Calendar,"EndOfService",SimRNG.Expon(mean_st,2))
        else:
            Server.Free(1)

        # if Policy(X,Y):
        #     ReadmissionProb = p_l
        #     Interventions.Record(1)
        # else:
        #     ReadmissionProb = p_h
            
        if SimRNG.Uniform(0,1,4) < ReadmissionProb:
            # join the orbit
            Orbit.Record(Y)
            SimFunctions.Schedule(Calendar,"Return",SimRNG.Expon(mean_ttr, 3))
            
            return X-1, Y+1
        else:
            return X-1, Y

    def Return(X, Y):
        Orbit.Record(Y)
        Congestion.Record(max(X-server_num, 0))
        Customer = SimClasses.Entity()
        Queue.Add(Customer)
        Readmissions.Record(1)

        if Server.Busy < server_num:
            Server.Seize(1)
            SimFunctions.Schedule(Calendar,"EndOfService",SimRNG.Expon(mean_st,2))

        return X+1, Y-1

    # lists for plotting sample paths
    Y_list = [0] # content
    X_list = [0] # needy
    time_list = [0]

    # Initialize 
    SimFunctions.SimFunctionsInit(Calendar,TheQueues,TheCTStats,TheDTStats,TheResources)
    SimFunctions.Schedule(Calendar,"EndSimulation",RunLength)
    SimFunctions.Schedule(Calendar,"Arrival", NSPP(1, params))

    for i in range(Y):
        SimFunctions.Schedule(Calendar,"Return",SimRNG.Expon(mean_ttr,3))

    for j in range(X):
        Customer = SimClasses.Entity()
        Queue.Add(Customer)

        if Server.Busy < server_num:
            Server.Seize(1)
            SimFunctions.Schedule(Calendar,"EndOfService",SimRNG.Expon(mean_st,2))

    NextEvent = Calendar.Remove()
    SimClasses.Clock = NextEvent.EventTime
    if NextEvent.EventType == "Arrival":
        X = Arrival(X)
    elif NextEvent.EventType == "EndOfService":
        X, Y = EndOfService(X, Y)
    elif NextEvent.EventType == "Return":
        X, Y = Return(X, Y)
    
    Y_list.append(Y) # content
    X_list.append(X) # needy
    time_list.append(SimClasses.Clock)
    
    while NextEvent.EventType != "EndSimulation":
        NextEvent = Calendar.Remove()
        SimClasses.Clock = NextEvent.EventTime
        if NextEvent.EventType == "Arrival":
            X = Arrival(X)
        elif NextEvent.EventType == "EndOfService":
            X, Y = EndOfService(X, Y)
        elif NextEvent.EventType == "Return":
            X, Y = Return(X, Y)

        # print(f"Progress: {SimClasses.Clock/RunLength*100:.2f}%")
        Y_list.append(Y)
        X_list.append(X)
        time_list.append(SimClasses.Clock)

    # POST SIMULATION RECORD-KEEPING
    Q = np.array(X_list) + np.array(Y_list)
    # plt.plot(time_list, Q, drawstyle='steps-post', linewidth=0.5, alpha=1)
    # # plt.plot(time_list, X_list, drawstyle='steps-post')
    # # plt.plot(time_list, Y_list, drawstyle='steps-post')
    # plt.ylabel('Total number of patients')
    # plt.xlabel(r'$t$')
    # plt.tight_layout()
    # plt.show()
    
    # print (Wait.Mean(), Queue.Mean(), Queue.NumQueue(), Server.Mean(), Orbit.Mean())
    # print(hold_cost * Congestion.Area(), readmit_cost * Readmissions.N(), interv_cost * Interventions.N())
    
    return np.array(Y_list), np.array(X_list), np.array(time_list)