

class Debugger:
    def __init__(self):
        self.debugCount = 1


    def DEBUG(self, arg):
        print('-----------------------  DEBUG no. # ',self.debugCount ,' ------------------------------------------------------\n\n')
        print(arg)
        print('\n--------------------------------------------------------------------------------------------------------------')
        print('########################################################################################################################')
        print('########################################################################################################################')
        print('########################################################################################################################')
        print('########################################################################################################################\n\n')
        self.debugCount +=1