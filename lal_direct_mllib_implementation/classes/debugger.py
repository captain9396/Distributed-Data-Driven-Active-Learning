import datetime
from datetime import timedelta



class Debugger:
    def __init__(self):
        self.debugCount = 1
        self.items = []
        self.t1 = datetime.datetime.now()
        self.totalTime = 0


    def DEBUG(self, arg):
        print('-----------------------  DEBUG no. # ',self.debugCount ,' --------------------------------------------------------------------------------\n')
        print(arg)
        print('\n\n\n---------------------------------------------------------------------------------------------------------------------------------------')
        print('########################################################################################################################')
        print('########################################################################################################################')
        print('########################################################################################################################')
        print('########################################################################################################################\n\n')
        self.debugCount +=1

    def add(self, items):
        self.items = items
        for x in self.items:
            self.DEBUG(x)


    def TIMESTAMP(self, id):
        print('####################-------------------------------      ', id,'     ------------------------------------------############################')
        return self.getRunningTime()

    def getRunningTime(self):
        t2 = datetime.datetime.now()
        time_difference = t2 - self.t1
        time_difference_in_minutes = time_difference / timedelta(minutes=1)
        self.totalTime += time_difference_in_minutes * 60
        print('************************    Time elapsed = ', time_difference_in_minutes, ' minutes    |||    ',time_difference_in_minutes * 60 ,' seconds ************************')
        print('-------------------------------------------------------   TOTAL TIME = ' , self.totalTime)
        self.t1 = t2
        return time_difference_in_minutes * 60



