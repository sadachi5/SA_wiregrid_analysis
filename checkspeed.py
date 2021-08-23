import os;
import numpy as np;
from matplotlib import pyplot as plt;
from utils import colors;
import pickle;


indir = 'output_ver2/pkl/PB20.13.13/PB20.13.13_Comb01Ch01';

files = ['A1_0', 'A2_22.5', 'A3_45', 'A4_67.5', 'A5_90', 'A6_112.5', 'A7_135', 'A8_157.5']

def checkspeed(indir, outdir='out_checkspeed', outname='HWP_speed.png') : 
    if not os.path.isdir(outdir) :
        print('Create a new directory: {}'.format(outdir));
        os.makedirs(outdir);
        pass;


    i_figs = 3;
    j_figs = 2;
    fig, axs = plt.subplots(i_figs,j_figs);
    fig.set_size_inches(6*j_figs,6*i_figs);
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, right=0.95,bottom=0.15, top=0.95)

    time00 = 0.;
    for i, a in enumerate(files) : 
        file = open(indir+'/'+a+'deg.pkl','rb')
        data = np.array(pickle.load(file));
        time = np.array(pickle.load(file));
        angle = np.array(pickle.load(file));
        angle_deg = angle * 180./np.pi;
    
        dt = np.diff(time)
        def getseconds(dt) : return dt.total_seconds(); # timedelta --> sec
        v_getseconds = np.vectorize(getseconds);
        dt = v_getseconds(dt); # timedelta --> sec
        dangle=np.diff(angle)
        dangle = dangle/(2.*np.pi) # rad. --> revo.
        dangle[dangle<-0.5] += 1
        speed = np.divide(dangle, dt); # Hz = revo./sec
        print('dangle<0 counts = {}'.format( (dangle<0.).sum() ));
        ave_speed = dangle.sum()/dt.sum();
        print('dt', dt);
        print('speed', speed);
        print('{:10s} : {:10f} Hz'.format(a,ave_speed));

        if time00==0. : time00 = time[0];
        time0 = time - time[0];
        time0 = v_getseconds(time0); # timedelta --> sec

        time_period = [];
        time_offset_from_angle0 = 0.;
        pre_time  = 0.;
        pre_angle = 0.;
        k_start = -1;
        for k, (__time, __angle) in enumerate(zip(time0, angle)) :
            if pre_angle > __angle :
                __dtime = __time - pre_time;
                __dangle= __angle + 2.*np.pi - pre_angle ;
                time_offset_from_angle0 = pre_time + __dtime * (2.*np.pi-pre_angle)/__dangle; # linear interpolation
                if k_start < 0. : k_start = k;
                pass;
            __time_period = __time - time_offset_from_angle0;
            time_period.append(__time_period);
            pre_time = __time;
            pre_angle= __angle;
            pass;
        time_period = np.array(time_period[k_start:]); # ignore first several points

        max_time_period = np.max(time_period);
        max_angle_deg = 360.;
        axs[0][0].plot(time_period, speed[k_start-1:], label=a, c=colors[i], linestyle='', linewidth=0, marker='o', markersize=0.5);
        axs[0][0].legend();
        axs[0][0].grid(True);
        axs[0][0].set_xlabel('Time Period [sec]');
        axs[0][0].set_ylabel('HWP Speed [Hz]');
        axs[0][0].set_xlim(0.,max_time_period);

        axs[1][0].plot(time_period, angle_deg[k_start:], label=a, c=colors[i], linestyle='', linewidth=0, marker='o', markersize=0.5);
        axs[1][0].legend();
        axs[1][0].grid(True);
        axs[1][0].set_xlabel('Time Period [sec]');
        axs[1][0].set_ylabel('HWP Angle [deg.]');
        axs[1][0].set_xlim(0.,max_time_period);

        diff_angle_deg = angle_deg[k_start:]-ave_speed*360.*time_period;
        print('ave. diff. angle deg. = {}'.format(np.mean(diff_angle_deg)));
        axs[2][0].plot(time_period, diff_angle_deg, label=a, c=colors[i], linestyle='', linewidth=0., marker='o', markersize=0.5);
        if a==files[-1] : axs[2][0].plot([0.,np.max(time_period)], [0.,0.], c='k', linestyle='-', linewidth=1., marker='', markersize=0.);
        axs[2][0].legend();
        axs[2][0].grid(True);
        axs[2][0].set_xlabel('Time period [sec.]');
        axs[2][0].set_ylabel('Diff. HWP angle from expected [deg.]');
        axs[2][0].set_xlim(0.,max_time_period);

        axs[0][1].plot(angle_deg[1:], speed, label=a, c=colors[i], linestyle='', linewidth=0, marker='o', markersize=0.5);
        axs[0][1].legend();
        axs[0][1].grid(True);
        axs[0][1].set_xlabel('HWP Angle [deg.]');
        axs[0][1].set_ylabel('HWP Speed [Hz]');
        axs[0][1].set_xlim(0.,max_angle_deg);

        time_1sec = v_getseconds(time - time00) %  (1./ave_speed); # timedelta --> sec
        axs[1][1].plot([],[], label=a, c=colors[i], linestyle='', linewidth=0, marker='o', markersize=2.); # dummy for larger legend
        axs[1][1].plot(time_1sec, angle_deg, c=colors[i], linestyle='', linewidth=0, marker='o', markersize=0.5);
        axs[1][1].legend();
        axs[1][1].grid(True);
        axs[1][1].set_xlabel('Time One Period [sec]');
        axs[1][1].set_ylabel('HWP Angle [deg.]');
        axs[1][1].set_xlim(0.,max_time_period);

        #if i==0 :  for an in angle_deg : print(an);
        pass;

    fig.savefig('{}/{}'.format(outdir, outname));
    return 0;


if __name__=='__main__' :
    checkspeed(indir);
    pass;
    
