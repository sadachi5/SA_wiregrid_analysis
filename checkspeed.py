import numpy as np;
import pickle;


indir = 'output_ver2/pkl/PB20.13.13/PB20.13.13_Comb01Ch01';

files = ['A1_0', 'A2_22.5', 'A3_45', 'A4_67.5', 'A5_90', 'A6_112.5', 'A7_135', 'A8_157.5']

def checkspeed(indir) : 
    for a in files : 
        file = open(indir+'/'+a+'deg.pkl','rb')
        data = pickle.load(file);
        time = pickle.load(file);
        angle = pickle.load(file);
    
        dt = np.diff(time)
        dangle=np.diff(angle)
        dangle = dangle/(2.*np.pi)
        dangle[dangle<-0.5] += 1
        speed = dangle.sum()/dt.sum().total_seconds()
        print('{:10s} : {:10f} Hz'.format(a,speed));
        pass;
    return 0;


if __name__=='__main__' :
    checkspeed(indir);
    pass;
    
