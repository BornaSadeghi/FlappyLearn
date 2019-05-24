import tensorflow as tf
import numpy as np
import colour, random
from geometry import inRect, vertRectCollision
from pygame import *
init()

dataFile = "flappy_learn.dat"

SIZE = 600,600
screen = display.set_mode(SIZE)
clock = time.Clock()

BLACK = 0,0,0
WHITE = 255,255,255
GREEN = 0,255,0
DARKGREEN = 0,150,0

GRAVITY = 0.98

birdX = 200
birdStartY = SIZE[1]//4
birdSize = 50
birdJumpH = 15

pipeWidth = 100
pipeSpeed = 3
pipeSpacing = 300 # space between pipe sets
gapSize = 220 # space between top and bottom pipes


class Bird():
    def __init__(self, auto=False, x=birdX, y=birdStartY):
        self.x, self.y = x,y
        self.rect = x,y,birdSize,birdSize
        self.yV = 0
        self.jumpH = birdJumpH
        self.colour = colour.randColour(False)
        
        self.jampThisFrame = False # for recording training data
        self.score = 0
        
        self.autonomous = auto # does an AI play for this bird?
        self.nn = True # is the AI the neural network? if not, it's the algorithm
        if auto and self.nn:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(100, activation="relu", input_dim=2),
                tf.keras.layers.Dense(40),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            if data != [] and labels != []:
                self.model.fit(data, labels, epochs=100)
            
    def update(self):
        self.yV -= GRAVITY
        self.y -= self.yV
        
        if self.yV < -self.jumpH: # terminal velocity
            self.yV = -self.jumpH
        
        if self.y > SIZE[1]-birdSize:
            self.y = SIZE[1]-birdSize
            self.yV = 0
        elif self.y < 0:
            self.y = 0
            self.yV = 0
            
        self.rect = self.x,self.y,birdSize,birdSize
        self.jampThisFrame = False
        
        if self.autonomous:
            if self.nn:
                data = np.asarray([normalizeData(self.getData())[:2]])
                prediction = self.model.predict(data)[0]
                  
#                 print(data, prediction)
              
                print(prediction[0])
                if prediction[0] > 0.5:
                    self.jump()
            elif shouldJump(self.getData()[:2]):
                self.jump()
    def jump(self):
        self.yV = self.jumpH
        self.jampThisFrame = True
        
    def getData(self): # for neural network
        nxt = nextPipe(self.x)
        bottom = nxt.bottom[1]
        return self.y, bottom, int(self.jampThisFrame)
    
    def draw(self):
        draw.rect(screen, self.colour, self.rect)


class Pipe:
    def __init__(self):
        self.x = SIZE[0]
        self.openingHeight = random.randrange(80, SIZE[1]-gapSize-80)
        self.addPoint = True
        
        self.top = Rect(self.x, 0, pipeWidth, self.openingHeight)
        self.bottom = Rect(self.x, self.openingHeight+gapSize, pipeWidth, SIZE[1]-self.openingHeight-gapSize)
        self.colour = GREEN
    def update(self):
        self.x -= pipeSpeed
        if self.x + pipeWidth < 0: # delete when off screen
            pipes.remove(self)
            
        if self == nextPipe(player.x):
            self.colour = DARKGREEN
        else:
            self.colour = GREEN
            
        self.top[0], self.bottom[0] = self.x, self.x
        
        for bird in birds:
            if self.x < bird.x-birdSize and self.addPoint:
                bird.score += 1
                self.addPoint = False
            elif vertRectCollision(bird.rect, self.top) or vertRectCollision(bird.rect, self.bottom):
                birds.remove(bird)
            
    def draw(self):
        draw.rect(screen, self.colour, self.top)
        draw.rect(screen, self.colour, self.bottom)

class SimpleText:
    def __init__(self, rect, text, fontSize=14, colour=(0,0,0)):
        self.rect = rect
        self.font = font.SysFont("lucida console", fontSize) # initialize font
        self.text = text
        self.textImg = self.font.render (text,False,colour)
        self.colour = colour
        
    def update (self, newText=""):
        self.text = newText
        self.textImg = self.font.render (newText,1, self.colour)  
    def draw(self):
        screen.blit(self.textImg, self.rect)

scoreCounter = SimpleText((0,0,100,100), "0", 64, WHITE)

def reset():
    global birds, pipes
    player.score = 0
    player.y = birdStartY
    player.yV = 0
    birds = [player]
    pipes = [Pipe()]

def nextPipe(x):
    for pipe in pipes:
        if pipe.x + pipeWidth - x < 0:
            continue
        else:
            return pipe
            
def writeToFile(string):

    f = open(dataFile, 'a')
    f.write(string)
    f.close()





# hard-coded AI: should the bird jump?

def shouldJump(data):
#     y, yV, yTop, yBottom, xDist = data
    y, yBottom = data
    
    if y+birdSize+20 >= yBottom:
        return True
    else:
        return False

def saveBirdData(bird):
    y, yBottom, label = normalizeData(bird.getData())
    string = str(y) + " " + str(yBottom) + " " + str(label)
    writeToFile(string.rstrip() + '\n')
    
def loadData():
    data, labels = [],[]
    with open(dataFile) as f:
        lines = f.readlines()
    lines = [line.rstrip('\n').split(' ') for line in lines]
        
    for line in lines:
        data.append([float(line[0]), float(line[1])])
        labels.append(int(line[-1]))
    return np.asarray(data), np.asarray(labels)

def normalizeData(rawData):
    y, yBottom, label = rawData
    return y/(SIZE[1]-birdSize), yBottom/SIZE[1], label

def events ():
    global run, recordData, player
    for e in event.get():
        if e.type == KEYDOWN:
            if e.key == K_SPACE:
                player.jump()
            elif e.key == K_1:
                recordData = not recordData
                print("Recording data..." if recordData else "Stopped recording data")
            elif e.key == K_2:
                player.autonomous = not player.autonomous
                print("AI is now playing" if player.autonomous else "Human is now playing")
            elif e.key == K_3:
                player = Bird(player.autonomous)
                reset()
        elif e.type == QUIT:
            run = False


data, labels = loadData()

player = Bird(True)
birds = [player]
pipes = [Pipe()]        

recordData = False # record training data to file
frameMax = 20
frameCount = 0

run = True
while run:
    screen.fill(BLACK)
    frameCount += 1
    
    for pipe in pipes:
        pipe.update()
        
    if pipes[-1].x < SIZE[0]-pipeSpacing:
        pipes.append(Pipe())
        
    for pipe in pipes:
        pipe.draw()
    
    if recordData:
        if player.jampThisFrame:
            saveBirdData(player)
        elif frameCount >= frameMax:
            saveBirdData(player)
            frameCount = 0
    
    for bird in birds:
        bird.update()
        bird.draw()
    
    scoreCounter.update(str(player.score))
    scoreCounter.draw()
    
    if birds == []:
        reset()
    
    events()
    display.update()
    clock.tick(60)
quit()