import numpy as np
import gym
import os,sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)




class MMSDENV():
    def __init__(self):
        self.num_of_grid = 3        #assume that every grid has a battery
        self.alpha = 0.5
        self.a_ug = 0.01
        self.b_ug = 0.1
        self.c_ug = 10
        self.lambda_ub = 5.5
        self.lambda_lb = 1.5
        self.delta_sn = 25
        self.sn_ub1 = 250
        self.sn_ub2 = 200
        self.pgn_lb = [-300.0, -300.0, -300.0]
        self.pgn_ub = [300.0, 300.0, 300.0]
        self.sn = np.zeros((1, self.num_of_grid))
        self.sn = self.sn.squeeze()
        self.sn[0] = self.sn_ub1/2.0 + (np.random.rand() * 10.0)
        self.sn[1] = self.sn_ub2/2.0 + (np.random.rand() * 10.0)
        self.vn = [[],[],[]]
        self.base_load = [[],[],[]]
        self.t_max = 12
        self.LoadData("vn",self.vn)
        self.LoadData("base_load",self.base_load)
        self.base_load0 = [244, 264, 114]
        self.omega = np.array([3.4442,3.7174,1.3810])
        self.lamb = 3.5#np.random.rand()*(self.lambda_ub - self.lambda_lb)  #price of power
        self.r1 = 0.0
        self.r2 = 0.0
        self.r3 = 0.0
        self.pgn = [135.0,163.0,60.0]
        self.t = 0
        self.check = -1
        self.ob1 = []
        self.ob2 = []
        self.ob3 = []
        self.state = np.array((self.t_max+1, 4))
        # self.shape_of_state = 5
        # self.shape_of_action = 3
        # self.shape_of_obs = 5
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]),high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf]),shape=None,dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf]),high=np.array([np.inf,np.inf,np.inf]),shape=None,dtype=np.float64)
        self.spec = None
        self.shape = 1
        self.it = -1
        self.cumlamb = 0.0
        self.cumpgn0 = 0.0
        self.cumpgn1 = 0.0
        self.cumpgn2 = 0.0

    def LoadData(self,fileName, container):
        with open("../environments/mmsdData/" + fileName + ".txt", "r") as f:
            lines = f.read()
            lines = lines.split(",	")
            ind = 0
            for i in range(len(lines)):
                temp = lines[i].split("\n")
                container[ind].append(float(temp[0]))
                if (i+1)% 49 == 0:
                    ind+=1


    def close(self):
        pass

    def h1(self):
        return 0.01*self.lamb*self.lamb-0.12*self.lamb+0.26

    def h2(self):
        return -0.01*self.lamb*self.lamb + 0.13

    def h3(self):
        return -0.01*self.lamb*self.lamb + 0.02*self.lamb + 0.08

    def Updn(self, sig, pdn):
        if  0<=pdn[sig]<=(self.omega[sig]/self.alpha):
            return self.omega[sig]*pdn[sig] - (self.alpha/2.0)*pdn[sig]*pdn[sig]
        else:
            return self.omega[sig]/self.alpha

    def Upgn(self):
        summ =0
        for i in range(self.num_of_grid):
            summ += self.pgn[i]
        return self.lamb*summ - self.a_ug*summ*summ
                            #NEXT STEP,increase the a_ug
    def reset(self):
        self.lamb = 3.5#np.random.rand()*(self.lambda_ub - self.lambda_lb)
        self.it = -1
        self.r1 = 0.0
        self.r2 = 0.0
        self.r3 = 0.0
        self.t = 0
        self.cumlamb = 0.0
        self.cumpgn0 = 0.0
        self.cumpgn1 = 0.0
        self.cumpgn2 = 0.0
        self.sn = np.zeros((1, self.num_of_grid))
        self.sn = self.sn.squeeze()
        self.sn[0] = self.sn_ub1/2.0 + (np.random.rand()*25.0)
        self.sn[1] = self.sn_ub2/2.0 + (np.random.rand() * 20.0)
        self.pgn[0] = 165.0
        self.pgn[1] = 163.0
        self.pgn[2] = 60.0
        self.cumlamb =3.0#+= self.lamb
        self.cumpgn0 =150.0#+= self.pgn[0]
        self.cumpgn1 =150.0#+= self.pgn[1]
        self.cumpgn2 =150.0#+= self.pgn[2]
        return np.array([(self.cumlamb)/66.0, (self.cumpgn0)/3600.0, (self.cumpgn1)/3600.0,(self.cumpgn2)/3600.0, self.t/self.t_max

                         ])

    def seed(self, sd):
        np.random.seed(sd)

    def step(self, a,task_id=0, sgenv1_coef=1.0, sgenv2_coef=1.0):
        done = False
        self.t += 1
        a = np.array(a)
        a = a.squeeze()
        if self.t == 1:
            pass
        a[2] = (a[2] - (-2.5)) / 2.5
        a[0] = (a[0] - (-2.5)) / 2.5
        a[1] = (a[1] - (-2.5)) / 2.5
        a[2] = max(-0.0, a[2])
        a[2] = min(2.0, a[2])
        a[0] = max(0.0, a[0])
        a[0] = min(2.0, a[0])
        a[1] = max(0.0, a[1])
        a[1] = min(2.0, a[1])
        a[2] = 1.5 + a[2] * 2.0
        a[0] = a[0]* (self.base_load[0][self.t]*float(sgenv1_coef)-self.vn[0][self.t])
        a[1] = a[1] * (self.base_load[1][self.t]*float(sgenv2_coef)-self.vn[1][self.t])
        self.lamb = a[2]
        pdn = [0.0,0.0,0.0]
        pdn[0] = (1.0 + self.h1()) * self.base_load[0][int(self.t)]
        pdn[1] = (1.0 + self.h2()) * self.base_load[1][int(self.t)]
        pdn[2] = (1.0 + self.h3()) * self.base_load[2][int(self.t)]
        for i in range(self.num_of_grid-1):
            self.pgn[i] = a[i]
        temp1 = self.sn[0]
        temp2 = self.sn[1]
        for i in range(self.num_of_grid-1):
            self.sn[i] = self.sn[i] + self.pgn[i] - pdn[i]+self.vn[i][self.t]
        delta1 = abs(temp1 - self.sn[0])
        delta2 = abs(temp2 - self.sn[1])
        self.pgn[self.num_of_grid-1] = pdn[self.num_of_grid-1]-self.vn[self.num_of_grid-1][self.t]
        reward1 =  0                                                 #Ud
        for i in range(self.num_of_grid):
            reward1 += (self.Updn(i, pdn) - self.lamb*pdn[i])
        reward2 = 0                                                  #Ug
        reward2 += self.Upgn()
        reward3 = 0                                                 #Sn0
        add1 = max(0.0, self.sn[0]-self.sn_ub1)+max(0.0,self.sn_ub1/2.0-self.sn[0])
        add2 = 0
        add2 += max(0.0, self.sn[1]-self.sn_ub2)+max(0.0,self.sn_ub2/2.0-self.sn[1])
        for i in range(self.num_of_grid-1):
            reward3 += self.sn[i]
        self.sn[0] = min(self.sn[0], self.sn_ub1)
        self.sn[0] = max(self.sn[0], self.sn_ub1 / 2.0)
        self.sn[1] = min(self.sn[1], self.sn_ub2)
        self.sn[1] = max(self.sn[1], self.sn_ub2 / 2.0)
        if add1 > 1.0:
            reward1 -= max(1.0,add1*95.0)
            reward2 -= max(1.0,add1*200.0)
            reward3 -=max(1.0, add1*3.8)
        if add2 > 1.0 :
            reward1 -= max(1.0,add2*95.0)
            reward2 -= max(1.0,add2*200.0)
            reward3 -= max(1.0, add2*3.8)
        if delta1 > 26.0 :
            reward1 -= max(1.0,(delta1-25.0)*95.0)
            reward2 -= max(1.0,(delta1-25.0)*200.0)
            reward3 -= max(1.0, (delta1-25.0)*3.8)
        if delta2 > 21.0 :
            reward1 -= max(1.0,(delta2-20.0)*95.0)
            reward2 -= max(1.0,(delta2-20.0)*200.0)
            reward3 -= max(1.0,(delta2-20.0)*3.8)
        if self.lamb > self.lambda_ub or self.lamb < self.lambda_lb:
            reward1 -= 25.0*(max(max(0.1,self.lamb-self.lambda_ub),max(0.1,self.lambda_lb-self.lamb)))
            reward2 -= 25.0*(max(max(0.1,self.lamb-self.lambda_ub),max(0.1,self.lambda_lb-self.lamb)))
            reward3 -= 25.0*(max(max(0.1,self.lamb-self.lambda_ub),max(0.1,self.lambda_lb-self.lamb)))
        self.lamb = min(self.lamb, self.lambda_ub)
        self.lamb = max(self.lamb, self.lambda_lb)
        self.cumlamb += self.lamb
        self.cumpgn0 += self.pgn[0]
        self.cumpgn1 += self.pgn[1]
        self.cumpgn2 += self.pgn[2]
        re1 = self.cumlamb
        re2 = self.cumpgn0
        re3 = self.cumpgn1
        re4 = self.cumpgn2
        re5 = self.t
        self.r1= self.sn[0]
        self.r2 = self.sn[1]
        self.r3 = self.lamb
        r4 = np.sum(self.pgn)
        r5 = self.lamb*r4-0.01*r4*r4
        if self.t >= self.t_max:
            done = True
        reward1 = -reward1
        reward2 = -reward2
        reward1 = max(0.0,(20000.0-reward1)/44.4)
        reward2 = max(0.0,(20000.0-reward2)/44.4)
        return np.array([(re1)/66.0, (re2)/3600.0, (re3)/3600.0,(re4)/3600.0, re5/self.t_max
                        ]), 0., done, {'obj': np.array([reward1/(450.0), reward2/(450.0), reward3/450.0]),'fail':np.array([self.r1,self.r2,self.r3]),'time':self.t,'origin':np.array([r4,r5])}


class sgenv2():
    def __init__(self):
        self.num_of_grid = 3  # assume that every grid has a battery
        self.alpha = 0.5
        self.a_ug = 0.01
        self.b_ug = 0.1
        self.c_ug = 10
        self.lambda_ub = 5.5
        self.lambda_lb = 1.5
        self.delta_sn = 25
        # self.sn_ub1 = 250
        # self.sn_ub2 = 200
        self.sn_ub1 = 250
        self.sn_ub2 = 200
        self.pgn_lb = [-300.0, -300.0, -300.0]
        self.pgn_ub = [300.0, 300.0, 300.0]
        self.sn = np.zeros((1, self.num_of_grid))
        self.sn = self.sn.squeeze()
        self.sn[0] = self.sn_ub1 / 2.0 + (np.random.rand() * 10.0)
        self.sn[1] = self.sn_ub2 / 2.0 + (np.random.rand() * 10.0)
        self.vn = [
            [54.0692993150888, 54.2941126030197, 54.7219676760005, 54.5772014249064, 54.362494784636, 54.0339945316526,
             54.0280891718406, 54.673909534658, 54.0235257745915, 54.0342664414056, 54.4116994810217, 54.159775624889,
             54.6864909290982, 54.462590973899, 54.0674115798487, 68.3961036647798, 68.4797462131965, 68.3278703495783,
             68.0178558392871, 68.4245646529344, 68.4669966238788, 68.3393675774289, 68.3788700652892, 68.3715662340625,
             68.1961135097671, 68.3277389450888, 68.0855933439058, 68.3530230440098, 68.0159164231887, 68.1384614924805,
             68.0230856953156, 68.0485658906179, 68.4117289141637, 68.3474143114879, 68.1585497400304, 70.0452442537222,
             69.0378906885532, 69.4826187956221, 69.4197143028023, 69.8420684669639, 69.8747198912508, 69.2055598650098,
             69.5387408353671, 69.490144820782, 69.7109443111224, 69.7803013139439, 69.8301553501806, 69.3036275846984,
             69.747672944539
             ],
            [130.413882395231, 131.004865917766, 131.05720278213, 130.601963137621, 130.048307249781, 130.791537128001,
             130.497678698824, 130.931425672611, 130.298479528833, 130.893885861449, 130.592851538002, 130.361107932812,
             130.130916096957, 140.887787427562, 140.764164658989, 140.641587332849, 140.644788201668, 140.677252476237,
             140.852215176795, 140.676284653691, 140.844285447821, 140.673057490618, 140.878779086956, 140.704995129795,
             140.658978575129, 141.251083857976, 141.616044676147, 141.473288848903, 141.351659507063, 141.830828627896,
             141.585264091153, 141.549723608291, 141.91719366383, 129.570804905898, 127.213998854446, 127.231354528608,
             129.097770765123, 128.160891796374, 130.620728552185, 130.730249406667, 128.346012234955, 127.10416384949,
             126.329946578854, 130.350468957631, 128.155881695639, 128.653046794709, 130.940489652494, 129.314386778006,
             130.189088459034
             ], [19.5588569081368, 19.462243008409, 19.5057066271012, 19.4331297459, 19.5203963882803, 19.452594256908,
                 19.5308158196954, 19.537842900628, 19.5496303185647, 19.4901083197005, 19.4167642755994,
                 19.4457953937434, 19.5826674723003, 19.4304756037938, 19.5651633954979, 19.507668487052,
                 19.5992269433254, 19.4156351057506, 19.4885356539551, 19.4213305540361, 2.2133707472117,
                 7.57404834484922, 3.26050139761558, 3.02310196434078, 2.73530964996434, 7.1271592651389,
                 5.36121716504618, 6.14472574403634, 3.11961651074388, 5.18408256620415, 119.91064759443,
                 119.181847028303, 119.263802916522, 119.145538980385, 119.136068558709, 119.86929220764,
                 119.579704587366, 119.549860201836, 119.144954798224, 119.853031117722, 119.622055131485,
                 119.350952380892, 119.513249539867, 119.401808033752, 119.075966691691, 119.239916153554,
                 119.123318934835, 119.183907788282, 119.239952525665
                 ]]
        self.base_load0 = [244, 264, 114]
        self.omega = self.base_load0
        self.omega = np.array(self.omega)
        self.omega = self.omega.squeeze()
        self.omega[0] = 3.4442
        self.omega[1] = 3.7174
        self.omega[2] = 1.3810
        self.base_load = [
            [244, 185, 161, 152, 149, 169, 182, 208, 254, 298, 351, 355, 435, 497, 460, 389, 396, 453, 467, 443, 410,
             361, 322, 290, 263, 184, 148, 144, 144, 158, 175, 194, 247, 297, 338, 378, 415, 447, 438, 413, 388, 441,
             471, 448, 400, 370, 342, 312, 269,
             ],
            [264, 266, 260, 266, 260, 266, 258, 260, 410, 610, 628, 658, 670, 616, 624, 644, 680, 648, 600, 576, 550,
             544, 432, 354, 354, 262, 260, 260, 262, 258, 262, 264, 414, 594, 606, 642, 648, 604, 610, 612, 638, 634,
             602, 576, 568, 540, 428, 354, 352,
             ],
            [114, 72, 57, 53, 53, 52, 70, 81, 71, 79, 80, 80, 114, 118, 91, 86, 88, 101, 123, 145, 134, 138, 147, 150,
             110, 69, 57, 57, 53, 54, 71, 72, 76, 90, 81, 83, 95, 104, 79, 76, 87, 100, 114, 137, 125, 128, 148, 148,
             115,
             ]]
        self.lamb = 3.5  # np.random.rand()*(self.lambda_ub - self.lambda_lb)  #price of power
        # self.pdn = self.base_load0
        self.r1 = 0.0
        self.r2 = 0.0
        self.r3 = 0.0
        self.pgn = [0.0, 0.0, 0.0]
        self.pgn[0] = 135.0
        self.pgn[1] = 163.0
        self.pgn[2] = 60.0
        # self.pgn[0] = (1.0+self.h1())*self.base_load[0][1]-self.vn[0][1]
        # self.pgn[1] = (1.0+self.h2())*self.base_load[1][1]-self.vn[1][1]
        # self.pgn[2] = (1.0+self.h3())*self.base_load[2][1]-self.vn[2][1]
        # print("init",self.lamb,self.sn[0],self.sn[1],self.pgn[0],self.pgn[1])
        self.t_max = 12
        self.t = 0
        self.check = -1
        self.ob1 = []
        self.ob2 = []
        self.ob3 = []
        self.state = np.array((self.t_max + 1, 4))
        self.shape_of_state = 5
        self.shape_of_action = 3
        self.shape_of_obs = 5
        self.observation_space = 5
        self.action_space = 3
        self.shape = 1
        self.it = -1
        self.cumlamb = 0.0
        self.cumpgn0 = 0.0
        self.cumpgn1 = 0.0
        self.cumpgn2 = 0.0

    def h1(self):
        return 0.01 * self.lamb * self.lamb - 0.12 * self.lamb + 0.26

    def h2(self):
        return -0.01 * self.lamb * self.lamb + 0.13

    def h3(self):
        return -0.01 * self.lamb * self.lamb + 0.02 * self.lamb + 0.08

    def Updn(self, sig, pdn):
        # print(self.omega)
        if 0 <= pdn[sig] <= (self.omega[sig] / self.alpha):
            return self.omega[sig] * pdn[sig] - (self.alpha / 2.0) * pdn[sig] * pdn[sig]
        else:
            return self.omega[sig] / self.alpha

    def Upgn(self):
        summ = 0
        for i in range(self.num_of_grid):
            summ += self.pgn[i]
        return self.lamb * summ - self.a_ug * summ * summ
        # NEXT STEP,increase the a_ug

    def reset(self):
        # self.pdn = self.base_load0
        self.lamb = 3.5  # np.random.rand()*(self.lambda_ub - self.lambda_lb)
        # self.pgn = self.base_load0
        self.it = -1
        self.r1 = 0.0
        self.r2 = 0.0
        self.r3 = 0.0
        self.t = 0
        self.cumlamb = 0.0
        self.cumpgn0 = 0.0
        self.cumpgn1 = 0.0
        self.cumpgn2 = 0.0
        self.sn = np.zeros((1, self.num_of_grid))
        self.sn = self.sn.squeeze()
        self.sn[0] = self.sn_ub1 / 2.0 + (np.random.rand() * 25.0)
        self.sn[1] = self.sn_ub2 / 2.0 + (np.random.rand() * 20.0)
        # self.pgn[0] = 100.0+np.random.rand()*80.0
        # self.pgn[1] = 100.0+np.random.rand()*80.0
        # self.pgn[2] = 30.0+np.random.rand()*80.0
        self.pgn[0] = 165.0
        self.pgn[1] = 163.0
        self.pgn[2] = 60.0
        self.cumlamb = 3.0  # += self.lamb
        self.cumpgn0 = 150.0  # += self.pgn[0]
        self.cumpgn1 = 150.0  # += self.pgn[1]
        self.cumpgn2 = 150.0  # += self.pgn[2]
        # self.pgn[0] = (1.0 + self.h1()) * self.base_load[0][1] - self.vn[0][1]
        # self.pgn[1] = (1.0 + self.h2()) * self.base_load[1][1] - self.vn[1][1]
        # self.pgn[2] = (1.0 + self.h3()) * self.base_load[2][1] - self.vn[2][1]
        return np.array(
            [(self.cumlamb) / 66.0, (self.cumpgn0) / 3600.0, (self.cumpgn1) / 3600.0, (self.cumpgn2) / 3600.0,
             self.t / self.t_max

             ])

    def seed(self, sd):
        np.random.seed(sd)

    def step(self, a, task_id=0, sgenv1_coef=1.0, sgenv2_coef=1.0):
        done = False
        self.t += 1
        a = np.array(a)
        a = a.squeeze()
        # if a[1] > 100.0:
        #     a[1]*=1.5
        # if a[2] > 100.0:
        #     a[2]*=1.5
        if self.t == 1:
            pass
            # self.lamb = 3.0+np.random.rand()*1.0
            # self.lamb = 3.2
            # a[0]=0.0
            # a[1] = 0.0
            # a[2] = 0.0
        a[2] = (a[2] - (-2.5)) / 2.5
        a[0] = (a[0] - (-2.5)) / 2.5
        a[1] = (a[1] - (-2.5)) / 2.5
        a[2] = max(-0.0, a[2])
        a[2] = min(2.0, a[2])
        a[0] = max(0.0, a[0])
        a[0] = min(2.0, a[0])
        a[1] = max(0.0, a[1])
        a[1] = min(2.0, a[1])
        a[2] = 1.5 + a[2] * 2.0
        # if a[1] > 0.95:
        #    a[1]*=1.1
        # if a[2] > 0.95:
        #    a[2]*=1.1
        a[0] = a[0] * (self.base_load[0][self.t] * float(sgenv1_coef) - self.vn[0][self.t])
        a[1] = a[1] * (self.base_load[1][self.t] * float(sgenv2_coef) - self.vn[1][self.t])
        # if a[1] > 250.0:
        #    a[1]*=1.2
        # if a[2] >250.0:
        #    a[2]*=1.2
        # if self.t == 1:
        #   a[0]=3.0
        #   a[1]=160.0#150.0+ np.random.rand()*20.0
        #   a[2]=160.0#150.0+np.random.rand()*20.0
        #     print(a)
        # print("lambda",self.t,self.lamb)
        self.lamb = a[2]
        pdn = [0.0, 0.0, 0.0]
        pdn[0] = (1.0 + self.h1()) * self.base_load[0][int(self.t)]
        pdn[1] = (1.0 + self.h2()) * self.base_load[1][int(self.t)]
        pdn[2] = (1.0 + self.h3()) * self.base_load[2][int(self.t)]
        for i in range(self.num_of_grid - 1):
            self.pgn[i] = a[i]
            # self.pgn[i] = max(pdn[i], self.pgn[i])
        # delta0 =  max(self.pgn[0] - pdn[0], self.delta_sn)
        # delta1 =  max(self.pgn[1] - pdn[1],self.delta_sn)
        # delta2 = max(self.pgn[2] - pdn[2], self.delta_sn)
        temp1 = self.sn[0]
        temp2 = self.sn[1]
        for i in range(self.num_of_grid - 1):
            self.sn[i] = self.sn[i] + self.pgn[i] - pdn[i] + self.vn[i][self.t]
        # print(self.t,self.lamb, self.sn[0], self.sn[1], a)
        # if iteration == self.it + 1:
        # if task_id == 2:
        #    print(self.t,self.lamb, a[1],a[2], self.sn[0],self.sn[1])
        delta1 = abs(temp1 - self.sn[0])
        delta2 = abs(temp2 - self.sn[1])
        self.pgn[self.num_of_grid - 1] = pdn[self.num_of_grid - 1] - self.vn[self.num_of_grid - 1][self.t]
        # print(self.t, self.pgn, self.lamb, a)
        print(self.t,self.lamb,self.pgn[0],self.pgn[1],self.pgn[2],pdn[0],pdn[1],pdn[2])
        reward1 = 0  # Ud
        # reward1 = self.Updn(self.check, pdn) - self.lamb*pdn[self.check]
        for i in range(self.num_of_grid):
            reward1 += (self.Updn(i, pdn) - self.lamb * pdn[i])
        # print(self.t,reward1)
        reward2 = 0  # Ug
        reward2 += self.Upgn()
        reward3 = 0  # Sn
        # if delta[0] > self.delta_sn or delta[1] > self.delta_sn or delta[2] > self.delta_sn:
        #    reward3 -= 100000
        add1 = 0
        add2 = 0
        add1 = max(0.0, self.sn[0] - self.sn_ub1) + max(0.0, self.sn_ub1 / 2.0 - self.sn[0])

        add2 += max(0.0, self.sn[1] - self.sn_ub2) + max(0.0, self.sn_ub2 / 2.0 - self.sn[1])
        for i in range(self.num_of_grid - 1):
            reward3 += self.sn[i]
        self.sn[0] = min(self.sn[0], self.sn_ub1)
        self.sn[0] = max(self.sn[0], self.sn_ub1 / 2.0)
        self.sn[1] = min(self.sn[1], self.sn_ub2)
        self.sn[1] = max(self.sn[1], self.sn_ub2 / 2.0)
        # if reward1 > -1.0:

        #    reward1 = -1.0
        # wif reward2 > -1.0:
        #    reward2 = -1.0
        if add1 > 1.0:
            reward1 -= max(1.0, add1 * 95.0)
            reward2 -= max(1.0, add1 * 200.0)
            reward3 -= max(1.0, add1 * 3.8)
        if add2 > 1.0:
            reward1 -= max(1.0, add2 * 95.0)
            reward2 -= max(1.0, add2 * 200.0)
            reward3 -= max(1.0, add2 * 3.8)
        if delta1 > 26.0:
            reward1 -= max(1.0, (delta1 - 25.0) * 95.0)
            reward2 -= max(1.0, (delta1 - 25.0) * 200.0)
            reward3 -= max(1.0, (delta1 - 25.0) * 3.8)
        if delta2 > 21.0:
            reward1 -= max(1.0, (delta2 - 20.0) * 95.0)
            reward2 -= max(1.0, (delta2 - 20.0) * 200.0)
            reward3 -= max(1.0, (delta2 - 20.0) * 3.8)
        if self.lamb > self.lambda_ub or self.lamb < self.lambda_lb:
            reward1 -= 25.0 * (max(max(0.1, self.lamb - self.lambda_ub), max(0.1, self.lambda_lb - self.lamb)))
            reward2 -= 25.0 * (max(max(0.1, self.lamb - self.lambda_ub), max(0.1, self.lambda_lb - self.lamb)))
            reward3 -= 25.0 * (max(max(0.1, self.lamb - self.lambda_ub), max(0.1, self.lambda_lb - self.lamb)))

            # reward1 = -800000.0
            # reward2 = -800000.0
        self.lamb = min(self.lamb, self.lambda_ub)
        self.lamb = max(self.lamb, self.lambda_lb)
        # if delta0 > self.delta_sn or delta1 > self.delta_sn or delta2 > self.delta_sn:
        #    reward3 = 0
        self.cumlamb += self.lamb
        self.cumpgn0 += self.pgn[0]
        self.cumpgn1 += self.pgn[1]
        self.cumpgn2 += self.pgn[2]
        re1 = self.cumlamb
        re2 = self.cumpgn0
        re3 = self.cumpgn1
        re4 = self.cumpgn2
        re5 = self.t
        self.r1 = self.sn[0]
        self.r2 = self.sn[1]
        self.r3 = self.lamb
        r4 = np.sum(self.pgn)
        r5 = self.lamb * r4 - 0.01 * r4 * r4

        if self.t >= self.t_max:
            done = True
            # self.reset()
        # print(self.t,reward1,reward2)

        reward1 = -reward1
        reward2 = -reward2
        reward1 = max(0.0, (20000.0 - reward1) / 44.4)
        reward2 = max(0.0, (20000.0 - reward2) / 44.4)
        # reward1 = #1.0/reward1
        # reward2 = #1.0/reward2
        # reward1 = (1.0/reward1)*100000.0*200.0
        # reward2 = (1.0/reward2)*100.0*200.0
        # print(self.t,reward1,reward2,reward3)
        # reward1 = 800000.0-reward1
        # reward2 = 800000.0-reward2
        # self.r1 = self.sn[0]
        # self.r2 = self.sn[1]
        # self.r3 = self.lamb

        # reward3 = 0.0
        add = 0
        # print(reward1,reward2,reward3,self.t)
        add3 = 0  # = max(0.0,re1-self.lambda_ub*(self.t+1))+max(0.0,(self.t+1)*self.lambda_lb-re1)
        # print(self.t,reward1,reward2,reward3)
        return np.array([(re1) / 66.0, (re2) / 3600.0, (re3) / 3600.0, (re4) / 3600.0, re5 / self.t_max
                         ]), 0., done, {'obj': np.array([reward1 / (450.0), reward2 / (450.0), reward3 / 450.0]),
                                        'fail': np.array([self.r1, self.r2, self.r3]), 'time': self.t,
                                        'origin': np.array([r4, r5])}

