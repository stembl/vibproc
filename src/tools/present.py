## Tools to illustrate the data
import matplotlib.pyplot as plt
import scipy as sp

from data_features import *

def plot_hist(peaks):

    ## Histogram of the peak impacts from the vibration data
    sig3_max, sig3_min = sigma_calc(peaks, 3)


    pkx = peaks.iloc[:,1]
    pky = peaks.iloc[:,2]
    pkz = peaks.iloc[:,3]

    # Plot Results
    plt.figure('Histogram', figsize=(8,8))
    #plt.title('Histogram of Peak Impacts On Each Axis')

    # Combined results
    plt.subplot(2,2,1)
    plt.hist(pkz, 30, normed=0, facecolor='green', alpha=0.50, label='Z')
    plt.hist(pky, 30, normed=0, facecolor='red', alpha=0.50, label='Y')
    plt.hist(pkx, 30, normed=0, facecolor='blue', alpha=0.50, label='X')

    plt.xlabel('Peak Acceleration [G]')
    plt.ylabel('Probability')
    plt.legend(loc='upper right', fontsize = 'small')
    plt.xlim(xmin=0)

    plt.grid(True)

    # Z Axis
    plt.subplot(2,2,2)
    nz, binsz, patchesz = plt.hist(pkz, 30, normed=0, facecolor='green', alpha=0.50)
    plt.plot((sp.mean(pkz),sp.mean(pkz)), (0, max(nz)), 'k--', linewidth=2, label=('Mean (%.2f)' % sp.mean(pkz)))
    plt.plot((sig3_max[3],sig3_max[3]), (0, max(nz)), 'k-.', linewidth=2, label=('3$\sigma$ (%.2f - %.2f)' %(sig3_min[3], sig3_max[3])))
    plt.plot((sig3_min[3],sig3_min[3]), (0, max(nz)), 'k-.', linewidth=2)
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(xmin=0)
    plt.grid(True)

    # X Axis
    plt.subplot(2,2,3)
    nx, binsx, patchesx = plt.hist(pkx, 30, normed=0, facecolor='blue', alpha=0.50)
    plt.plot((sp.mean(pkx),sp.mean(pkx)), (0, max(nx)), 'k--', linewidth=2, label=('Mean (%.2f)' % sp.mean(pkx)))
    plt.plot((sig3_max[2],sig3_max[2]), (0, max(nx)), 'k-.', linewidth=2, label=('3$\sigma$ (%.2f - %.2f)' %(sig3_min[2], sig3_max[2])))
    plt.plot((sig3_min[2],sig3_min[2]), (0, max(nx)), 'k-.', linewidth=2)
    plt.legend(loc='upper right', fontsize='small')

    plt.xlabel('Peak Acceleration [G]')
    plt.ylabel('Probability')
    plt.xlim(xmin=0)
    plt.grid(True)

    # Y Axis
    plt.subplot(2,2,4)
    ny, binsy, patchesy = plt.hist(pky, 30, normed=0, facecolor='red', alpha=0.50)
    plt.plot((sp.mean(pky),sp.mean(pky)), (0, max(ny)), 'k--', linewidth=2, label=('Mean (%.2f)' % sp.mean(pky)))
    plt.plot((sig3_max[1],sig3_max[1]), (0, max(ny)), 'k-.', linewidth=2, label=('3$\sigma$ (%.2f - %.2f)' %(sig3_min[1], sig3_max[1])))
    plt.plot((sig3_min[1],sig3_min[1]), (0, max(ny)), 'k-.', linewidth=2)
    plt.legend(loc='upper right', fontsize='small')

    plt.xlabel('Peak Acceleration [G]')
    plt.xlim(xmin=0)
    plt.grid(True)

    #add a best fit line curve
    #y = mlab.normpdf(bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.savefig('hist_data')
    plt.show()

def plot_psd(avg_psd, ref):

    head = list(avg_psd.columns)

    ## Plot the average PSD response
    color = iter(plt.cm.tab10(sp.linspace(0,1,len(head)-1)))

    ## Print Average PSD Results
    plt.figure('PSD')
    plt.loglog(ref.Freq,ref.PSD, 'k', label='Input')
    for i in range(len(head)-1):
        c = next(color)
        plt.loglog(avg_psd[head[0]], avg_psd[head[i+1]], c=c, label=head[i+1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [G^2/Hz]')
    plt.title('PSD Response')
    plt.legend(loc='best')
    plt.axis([1,200,1e-6, 1e-1])

    plt.show()

def present_vib_data(data, ref):
    ## Print results together
    #  Need: trans, grms, avg_psd, peaks, 


    if (ref.shape[1] == 4):
        print("The input GRMS is :\n",
              "R1 : %.2f,  R2 : %.2f,  R3 : %.2f [G]" %((grms(ref.iloc[:,0],ref.iloc[:,1])),
                                                     (grms(ref.iloc[:,0],ref.iloc[:,2])),
                                                     (grms(ref.iloc[:,0],ref.iloc[:,3]))))

        print("The response GRMS is :\n",
            "R1 : %.2f,  R2 : %.2f,  R3 : %.2f [G]" %(grms(avg_psd.Freq, avg_psd.iloc[:,1]),
                                                   grms(avg_psd.Freq, avg_psd.iloc[:,2]),
                                                   grms(avg_psd.Freq, avg_psd.iloc[:,3])))
    else:
        print("The input GRMS is : %.2f" %(grms(ref.iloc[:,0],ref.iloc[:,1])))

        print("The response GRMS is : %.2f" %grms(avg_psd.Freq, avg_psd.iloc[:,1]))

    print("Out of %i events, the maximum impacts are \n" %(data.shape[0]),
          "R1 : %.2f,  R2 : %.2f,  R3 : %.2f [G]" %(max(peaks.iloc[:,1]), max(peaks.iloc[:,2]), max(peaks.iloc[:,2])))
    print('The average peak impacts by axis are \n',
         "R1 : %.2f,  R2 : %.2f,  R3 : %.2f [G]" %(sp.mean(peaks.iloc[:,1]), sp.mean(peaks.iloc[:,2]), sp.mean(peaks.iloc[:,3])))

    print('The 3 sigma values are \n',
          "R1 : %.2f,  R2 : %.2f,  R3 : %.2f [G]"
          %(sig3_max.iloc[:,1], sig3_max.iloc[:,2], sig3_max.iloc[:,3]))

    data_to_hist(peaks)

    # Pull reference curve for illustration
    ista_air = vib_profiles('ista air ride')

    plt.figure('PSD', figsize=(8,4))
    if trans.shape[1]>2:
        plt.loglog(ista_air.Freq, ista_air.PSD, 'k', label='ISTA Air (Ref)')
        plt.loglog(ref.Freq, ref.X, '--b', label='Input')
        plt.loglog(ref.Freq, ref.Y, '--r', label='')
        plt.loglog(ref.Freq, ref.Z, '--g', label='')
        plt.loglog(avg_psd.Freq, avg_psd.X, 'b', label='X')
        plt.loglog(avg_psd.Freq, avg_psd.Y, 'r', label='Y')
        plt.loglog(avg_psd.Freq, avg_psd.Z, 'g', label='Z')
    else:
        psdin = ref
        plt.figure()
        plt.loglog(ista_air.Freq, ista_air.PSD, '--k', label='ISTA Air (Ref)')
        plt.loglog(psdin.Freq, psdin.PSD, 'k', label='Input')
        plt.loglog(avg_psd.Freq, avg_psd.iloc[:,1], 'b', label='X')
        plt.loglog(avg_psd.Freq, avg_psd.iloc[:,2], 'r', label='Y')
        plt.loglog(avg_psd.Freq, avg_psd.iloc[:,3], 'g', label='Z')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [G^2/Hz]')
    plt.title('PSD Response')
    plt.legend(loc='best', fontsize='small')
    plt.axis([1,200,1e-6, 1e-1])
    plt.savefig('psd')
    plt.show()

    plt.figure('Trans', figsize=(8,4))
    if trans.shape[1]<=2:
        plt.loglog(trans.Freq, trans.R0, 'g', label='Z')
    if trans.shape[1]>2:
        plt.loglog(trans.Freq, trans.R0, 'b', label='X')
        plt.loglog(trans.Freq, trans.R1, 'r', label='Y')
        plt.loglog(trans.Freq, trans.R2, 'g', label='Z')
    plt.legend(loc='best', fontsize='small')
    plt.loglog([1, 200], [1, 1], 'k')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transmissibility')
    plt.title('Transmissibility of the Response')
    plt.axis([1,200,0.1,30])
    plt.savefig('trans')
    plt.show()
