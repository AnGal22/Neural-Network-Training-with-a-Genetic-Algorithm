import sys
import numpy

arg = {}
for i in range(1, len(sys.argv), 2):
    arg[sys.argv[i]] = sys.argv[i+1]

train = arg["--train"]
test = arg["--test"]
nn = arg["--nn"]
popsize = arg["--popsize"]
elitism = arg["--elitism"]
p = arg["--p"]
K = arg["--K"]
iter = arg["--iter"]


#ucitavamo training i test podate
with open(train, "r") as file:
    redovi = file.readlines()
    
with open(test, "r") as file:
    redovi_test = file.readlines()
#header razdvajam na znacajke i cilj
header = redovi[0].strip().split(",")
cilj = header[-1]
xi = header[:-1]

dimens = [] #dimenzije ovise o parametru nn
dimens.append(int(len(xi)))
if nn == "5s":
    dimens.append(5)
    dimens.append(1)
elif nn == "20s":
    dimens.append(20)
    dimens.append(1)
elif nn == "5s5s":
    dimens.append(5)
    dimens.append(5)
    dimens.append(1)

header_test = redovi_test[0].strip().split(",")

test_primjeri = []

primjeri = []


for red in redovi[1:]:#train i test stavljam primjere stavljam u rjecinik
    vrij = red.strip().split(",")
    primjer = {}
    for i in range(len(header)):
        primjer[header[i]] = float(vrij[i])
    primjeri.append(primjer)

for red in redovi_test[1:]:
    vrij = red.strip().split(",")
    primjer = {}
    for i in range(len(header_test)):
        primjer[header_test[i]] = float(vrij[i])
    test_primjeri.append(primjer)

#podatci u numpy matrice jer je inace presporo racunanje
X = numpy.array([[primjer[t] for t in xi] for primjer in primjeri])
y =  numpy.array([[primjer[cilj]] for primjer in primjeri])

X_test = numpy.array([[primjer[t] for t in xi] for primjer in test_primjeri])
y_test =  numpy.array([[primjer[cilj]] for primjer in test_primjeri])

class NeuralNet:
    def __init__(self, dimens):
        self.slojevi = []#inicijalizacija sa tezinama dobivenih pomocu normalne razdiobe
        for i in range(len(dimens) - 1):
            w = numpy.random.normal(0, 0.01, (dimens[i], dimens[i+1]))
            b = numpy.random.normal(0, 0.01, (dimens[i+1],))
            self.slojevi.append([w, b])

    def sigmoid(self, net):#sigmoidalna funkcija
        sig = 1/(1 + numpy.exp(-net))
        return sig

    def forward(self, x):#prolazi kroz sve slojeve
        for i in range(len(self.slojevi)):
            w, b = self.slojevi[i]
            net = x @ w + b
            if i < len(self.slojevi) - 1:
                x = self.sigmoid(net)
            else:
                x = net
        return x
    
    def mse(self, X, y):#srednje kvadratno odstupanje predikcije i prave vrijednosti
        pred_y = self.forward(X)
        err = numpy.mean((y - pred_y) ** 2)
        return err

    def getweights(self):
        return self.slojevi
    
    def setweights(self, novslojevi):
        self.slojevi = novslojevi


popul = []#populacija mreza
mse_popul = []
for i in range(int(popsize)):
    popul.append(NeuralNet(dimens))

for mreza in popul:#evaulacija
    mse_popul.append(mreza.mse(X, y))
for br in range(int(iter)):
    parovi = list(zip(mse_popul, popul))#stavlja greske mreze i mrezu u par
    parovi.sort(key=lambda x: x[0])#sortira mreze po pogreskama
    novpopul = [par[1] for par in parovi[:int(elitism)]]#selektira najbolje neuronske mreze s najmanjom pogrskom
    dobrote = []
    ukdobrote = 0
    udio = []
    for par in parovi:
        dobrote.append(1/par[0])#inverz MSE
        ukdobrote += 1/par[0]

    for i in range(len(parovi)):
        udio.append(dobrote[i]/ukdobrote)#udio pomocu koje selekcija bira parenta
    udio = numpy.array(udio).flatten()#potrebno flatten da udio bude jednodimennzijalan
    #print(udio.shape)

    while len(novpopul) < int(popsize):
        #selekcija po proporcionalnoj dobroti
        ind = numpy.random.choice(len(parovi), p=udio)
        roditelj1 = parovi[ind][1]
        ind = numpy.random.choice(len(parovi), p=udio)
        roditelj2 = parovi[ind][1]

        #krizanje po aritmetickoj sredini tezina
        slojevi1 = roditelj1.getweights()
        slojevi2 = roditelj2.getweights()
        novislojevi = []
        for (w1,b1), (w2,b2) in zip(slojevi1, slojevi2):
            nov_w = (w1 + w2)/2
            nov_b = (b1 + b2)/2
            novislojevi.append([nov_w, nov_b])

            #mutacija pomocu gaussovog suma s vjerovatnoscu p za svaku tezinu
        novislojevi_mut = []
        for w, b in novislojevi:
            #koristim maske jer ubrza matricno racunanje
            maska_w = numpy.random.random(w.shape) < float(p)
            w = w + maska_w * numpy.random.normal(0, float(K), w.shape)

            maska_b = numpy.random.random(b.shape) < float(p)
            b = b + maska_b * numpy.random.normal(0, float(K), b.shape)
            
            novislojevi_mut.append([w, b])

        dijete = NeuralNet(dimens)#kreiramo potomka i spremamo za iducu generaciju
        dijete.setweights(novislojevi_mut)
        novpopul.append(dijete)

    
    popul = novpopul#azuriram populaciju i evaulaciju
    mse_popul = []
    for mreza in popul:
        mse_popul.append(mreza.mse(X, y))
    
    if (br + 1) % 2000 == 0:#ispis svakih 2000 linija MSE najboljeg 
        print(f"[Train error @{br+1}]: {parovi[0][0]:.6f}")

naj = parovi[0][1]#test greske za najbolje modela nakon treniranja
print(f"[Test error]: {naj.mse(X_test, y_test):.6f}")