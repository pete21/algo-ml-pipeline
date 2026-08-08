import numpy as np
from backtesting.lib import Strategy


class Daytrading_strategy(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()

        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
#        self.set_trailing_sl(2)
    def next(self):
#        print(self.data.df['labeling_multi'].iloc[-1])
        if self.data.DaytradingExit[-1]:
            if self.position:      # daytrading
                self.position.close()
            return

        if not self.position:
            if self.data.y_pred[-1]>=1:
                self.buy(size=1, limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
                return
            if self.data.y_pred[-1]<=-1:
                self.sell(size=1, limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
                return

        # else:   # if there is position

        #     if self.position.pl_pct >= 0.002: # if profit percentage is greater than 0.15%, adjust stop-loss to sl/2 from current price (~ break even price)
        #         for trade in self.trades:
        #             if trade.is_long:
        #                 trade.sl = max(trade.sl or -np.inf, (1-self.data.sl[-1]/2)*self.data.Close[-1])
        #             elif trade.is_short:
        #                 trade.sl = min(trade.sl or np.inf, (1+self.data.sl[-1]/2)*self.data.Close[-1])

            # if self.position.pl_pct >= 0.0015: # if profit percentage is greater than 0.2%, open addon trade with half SL and half TP
            #     if self.data.y_pred[-1]>0.5 and self.position.size == 1:
            #         self.buy(size=1, limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1]/2)*self.data.Close[-1], tag=None)
            #     elif self.data.y_pred[-1]<-0.5 and self.position.size == -1:
            #         self.sell(size=1, limit=None, stop=None, tp=(1-self.data.tp[-1]/2)*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)



class Trailing_drawdown_strategy(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()

    def next(self):

        if self.position:
            if self.data.DaytradingExit[-1]:
                self.position.close()
                return
            for trade in self.trades:
                if trade.is_long:
                    trade.sl = max(trade.sl or -np.inf, self.data.High[-1] - self.data.sl[-1])
                elif trade.is_short:
                    trade.sl = min(trade.sl or np.inf, self.data.Low[-1] + self.data.sl[-1])
            return
        
        if self.data.y_pred[-1]>=1:
            # if self.position.is_short:
            #    self.position.close()
            #    return
            # if not self.position:
#                self.buy(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
            self.buy(size=1, limit=None, stop=None, sl=self.data.Close[-1] - self.data.sl[-1], tp=self.data.Close[-1] + self.data.tp[-1], tag=None)
            return

        if self.data.y_pred[-1]<=-1:
            # if self.position.is_long:
            #    self.position.close()
            #    return
            # if not self.position:
#                self.sell(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
            self.sell(size=1, limit=None, stop=None, tp=self.data.Close[-1] - self.data.tp[-1], sl=self.data.Close[-1] + self.data.sl[-1], tag=None)







# class TrailingStrategy(SignalStrategy):

#     __sl_amount = 100
#     def set_trailing_sl(self, sl_amount: float = 100):
#         """
#     Set the trailing stop loss as $n below the current price (for long positions)
#         Works for future bars only
#         """
#         self.__sl_amount = sl_amount


#     def init(self):
#         # In init() and in next() it is important to call the
#         # super method to properly initialize the parent classes
#         super().init()

#         # Set trailing stop-loss to 2x ATR using
#         # the method provided by `TrailingStrategy`
# #        self.set_trailing_sl(2)
#     def next(self):
# #        print(self.data.df['labeling_multi'].iloc[-1])

#         for trade in self.trades:
#             if trade.is_long:
#                 trade.sl = max(trade.sl or -np.inf, self.data.High[-1] - self.__sl_amount)
#             elif trade.is_short:
#                 trade.sl = min(trade.sl or np.inf, self.data.Low[-1] + self.__sl_amount)

# class Trailing_drawdown_strategy(TrailingStrategy):

#     def init(self, sl_amount: float = 100, tp_amount: float = 150):
#         # In init() and in next() it is important to call the
#         # super method to properly initialize the parent classes
#         super().init()
#         self.set_trailing_sl(sl_amount)

#     def next(self):
# #        print(self.data.df['labeling_multi'].iloc[-1])
#         super().next()

#         if self.position:
#             if self.data.DaytradingExit[-1]:
#                 self.position.close()
#                 return
#             return
        
#         if self.data.y_pred[-1]>=0.5:
#             # if self.position.is_short:
#             #    self.position.close()
#             #    return
#             # if not self.position:
# #                self.buy(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
#             self.buy(size=1, limit=None, stop=None, sl=self.data.Close[-1]-self.__sl_amount, tp=self.data.Close[-1]+self.__tp_amount, tag=None)

#         if self.data.y_pred[-1]<=-0.5:
#             # if self.position.is_long:
#             #    self.position.close()
#             #    return
#             # if not self.position:
# #                self.sell(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
#             self.sell(size=1, limit=None, stop=None, tp=self.data.Close[-1]-self.__tp_amount, sl=self.data.Close[-1]+self.__sl_amount, tag=None)
