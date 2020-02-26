class Stock:
    def __init__(self, highs, lows, volume, opens, closes, name, ticker, average):
        self.highs= highs;
        self.lows= lows;
        self.volume= volume;
        self.opens= opens;
        self.closes= closes;
        self.name= name;
        self.ticker= ticker;
        self.average= average;
    def getAverage(self):
        return self.average;
    def getTicker(self):
        return self.ticker;
    def getName(self):
        return self.name;
    def getHighs(self):
        return self.highs;
    def getLows(self):
        return self.lows;
    def getOpens(self):
        return self.opens;
    def getCloses(self):
        return self.closes;
    def getVolume(self):
        return self.volume;
    def setTicker(self, ticker):
        self.ticker= ticker;
    def setName(self, name):
        self.name= name;
    def setVolume(self, vol):
        self.volume= vol;
    def setHighs(self, high):
        self.highs= high;
    def setLows(self, lows):
        self.lows= lows;
    def setOpens(self, opens):
        self.opens= opens;
    def setCloses(self, closes):
        self.closes= closes;
    def getTwelveDayAverage(self):
        length= len(self.opens);
        total=0;
        if(length<12):
            for a in self.opens:
                total+=a;
            return total/length;
        else:
            for p in range(length-13, length):
                total+=self.opens[p];
            return total/12;
    def addSingleLow(self, low):
        self.lows.append(low);
    def addSingleHigh(self, high):
        self.highs.append(high);
    def addSingleVol(self, volume):
        self.volume.append(volume);
    def addSingleOpen(self, open):
        self.opens.append(open);
    def addSingleClose(self, close):
        self.closes.append(close);    
        
    
