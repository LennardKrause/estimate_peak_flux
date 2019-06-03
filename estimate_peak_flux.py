import numpy as np
import glob, argparse, os, re, sys
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy.special import erf
import logging

def init_argparser():
    parser = argparse.ArgumentParser(description = 'Find local maxima on frames, filter peaks and fit the resulting reflection profiles assuming a Gaussian model to estimate the peak flux.')
    parser.add_argument('-f', required=True,  default='',   metavar='path', type=str, dest='_FRAME', help='Specify the frame file path, use \'*\' and \'[0-9]\' as wildcards')
    parser.add_argument('-r', required=False, default=None, metavar='path', type=str, dest='_RAW', help='Specify the .raw file path, use \'*\' and \'[0-9]\' as wildcards')
    parser.add_argument('-o', required=False, default=None, metavar='path', type=str, dest='_OUTPATH', help='Specify the output path')
    parser.add_argument('-q', required=False, default=3,    metavar='int', type=int, dest='_QUEUE', help='Image queue, 2N+1 frames to monitor a profile, default: 3')
    parser.add_argument('-i', required=False, default=1e5,  metavar='int', type=int, dest='_MININT', help='Set the minimum peak intensity threshold for the peak search, default: 100000')
    parser.add_argument('-m', required=False, action='store_true',  dest='_USEMAT', help='Track peaks by using the maximum value in a 3x3 matrix around the peak position instead of the exact coordinates')
    parser.add_argument('-t', required=False, action='store_true',  dest='_TAILS', help='Fit Gaussian to tails (additionally)')
    parser.add_argument('-l', required=False, action='store_true',  dest='_LORENTZ', help='Fit Lorentzian (additionally)')
    parser.add_argument('-p', required=False, action='store_false', dest='_PLOT', help='Store plots of the fitted peaks in *.pdf (*framename-stem)')
    # print help if script is run without arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        raise SystemExit
    return parser.parse_args()

def read_sfrm(fname):
    #def chunkstring(string, length):
    #    '''
    #     return header as list of tuples
    #      - splits once at ':'
    #      - keys and values are stripped strings
    #      - values with more than 1 entry are un-splitted
    #    '''
    #    return list(tuple(map(lambda i: i.strip(), string[0+i:length+i].split(':', 1))) for i in range(0, len(string), length)) 
    #header_list = chunkstring(header, 80)
    with open(fname, 'rb') as f:
        # read the first 512 bytes
        # find keyword 'HDRBLKS' 
        header_0 = f.read(512).decode()
        # header consists of HDRBLKS x 512 byte blocks
        header_blocks = int(re.findall('\s*HDRBLKS\s*:\s*(\d+)', header_0)[0])
        # read the remaining header
        header = header_0 + f.read(header_blocks * 512 - 512).decode()
        # extract frame info:
        # - rows, cols (NROWS, NCOLS)
        # - bytes-per-pixel of image (NPIXELB)
        # - length of 16 and 32 bit overflow tables (NOVERFL)
        nrows = int(re.findall('\s*NROWS\s*:\s*(\d+)', header)[0])
        ncols = int(re.findall('\s*NCOLS\s*:\s*(\d+)', header)[0])
        npixb = int(re.findall('\s*NPIXELB\s*:\s*(\d+)', header)[0])
        nov16, nov32 = list(map(int, re.findall('\s*NOVERFL\s*:\s*-*\d+\s+(\d+)\s+(\d+)', header)[0]))
        # calculate the size of the image
        im_size = nrows * ncols * npixb
        # bytes-per-pixel to datatype
        bpp2dt = [None, np.uint8, np.uint16, None, np.uint32]
        # reshape data, set datatype to np.uint32
        data = np.fromstring(f.read(im_size), bpp2dt[npixb]).reshape((nrows, ncols)).astype(np.uint32)
        # read the 16 bit overflow table
        # table is padded to a multiple of 16 bytes
        read_16 = int(np.ceil(nov16 * 2 / 16)) * 16
        # read the table, trim the trailing zeros
        table_16 = np.trim_zeros(np.fromstring(f.read(read_16), np.uint16))
        # read the 32 bit overflow table
        # table is padded to a multiple of 16 bytes
        read_32 = int(np.ceil(nov32 * 4 / 16)) * 16
        # read the table, trim the trailing zeros
        table_32 = np.trim_zeros(np.fromstring(f.read(read_32), np.uint32))
        # assign values from 16 bit overflow table
        data[data == 255] = table_16
        # assign values from 32 bit overflow table
        data[data == 65535] = table_32
        return header, data

def find_peaks(frames, queue, thresh):
    # first iteration, find maxima and append to list
    # read the data, store relevant frame data in list
    local_max = []
    frame_data = []
    ind_to_store = []
    logging.info('\n >>> Hunting Peaks')
    for idx, fname in enumerate(frames):
        # read the data
        _, f = read_sfrm(fname)
        bname = os.path.basename(fname)
        # tell me where you are
        print(bname, end='\r')
        # save the data
        frame_data.append(f)
        # find position where data larger threshold
        pos = np.transpose(np.where(f >= thresh))
        # skip peaks whose tails are outside the frame list
        i_min = idx - queue
        i_max = idx + queue +1
        # did we find something?
        if len(pos) > 0 and i_max <= len(frames) and i_min >= 0:
            # add idices of the full peak range
            # to the do-not-delete list (ind_to_store)
            ind_to_store.extend(range(i_min, i_max))
            # save the peak position
            for p in pos:
                px, py = p
                peak = f[px,py]
                logging.info('{} > {:7} @ {:>4}x{:<4}'.format(bname, peak, px, py))
                local_max.append((bname, idx, px, py, peak))
        # forget unnecessarily stored data
        idx_del = i_min
        if idx_del >= 0 and idx_del not in ind_to_store:
            frame_data[idx_del] = None
    print()
    return frame_data, local_max

def filter_peaks(local_max, queue):
    logging.info('\n >>> Filtering Peaklist: {}'.format(len(local_max)))
    # filter local_max for overlapping spots
    # lx,ly,li,lm: last x, y, intensity, multiplicity
    lx,ly,li,lm = 0,0,0,1
    filtered = []
    reject = []
    local_max.sort(key = lambda v: v[2])
    for id in local_max:
        (bname, idx, px, py, peak) = id
        # track spot position in 3x3 matrix
        prefix = ' '
        if px in range(lx-1, lx+2) and py in range(ly-1, ly+2):
            lm += 1
            if peak > li:
                prefix = '+'
                filtered.pop()
                reject.pop()
            else:
                logging.info('-{}:{:7} @ {:>4}x{:<4}/{:3}'.format(bname, peak, px, py, lm))
                continue
        else:
            lm = 1
        reject.append(lm)
        filtered.append(id)
        lx, ly, li = px, py, peak
        logging.info('{}{}:{:7} @ {:>4}x{:<4}/{:3}'.format(prefix, bname, peak, px, py, lm))
    filtered = [n for i,n in enumerate(filtered) if reject[i] < queue]
    logging.info('\n >>> Filtered Peaklist: {}'.format(len(filtered)))
    [logging.info('{}{:5}{:7} @ {:>4}x{:<4}'.format(bname, idx, peak, px, py)) for (bname, idx, px, py, peak) in filtered]
    return filtered

def get_frame_info(fnam):
    # get run and frame number from file name
    # some_data_*run_*frame.sfrm
    # e.g. some_data_01_0001.sfrm
    try:
        bname, ext = os.path.splitext(fnam)
        _split = bname.split('_')
        fnum = int(_split.pop())
        rnum = int(_split.pop())
        stem = '_'.join(_split)
        return stem, rnum, fnum
    except (ValueError, IndexError):
        return None, None
    
def main():
            
    def I_gauss(x,I,mu,sig):
        return I*(1/2.*(1+erf(((x+dx/2.)-mu)/(sig*np.sqrt(2)))) - 1/2.*(1+erf(((x-dx/2.)-mu)/(sig*np.sqrt(2)))))
    
    def f_gauss(t,I,mu,sig):
        return I*(1/((sig/v)*np.sqrt(2*np.pi))*np.exp(-1/2.*(t-mu/v)**2/(sig/v)**2))
    
    def f_gauss_ang(t,I,mu,sig):
        return I*(1/((sig)*np.sqrt(2*np.pi))*np.exp(-1/2.*(t-mu)**2/(sig)**2))
    
    def I_lorentz(x,I,mu,gamma):
        return I*(1/np.pi*np.arctan2((x+dx/2.)-mu,gamma) + 1/2. - (1/np.pi*np.arctan2((x-dx/2.)-mu,gamma) + 1/2.))
        
    def f_lorentz(t,I,mu,gamma):
        return I*1/np.pi*(gamma/v)/((t-mu/v)**2+(gamma/v)**2)
    
    _ARGS = init_argparser()
    
    mpl.rcParams['figure.figsize'] = [12.60, 7.68]
    mpl.rcParams['savefig.dpi'] = 100
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.titlesize'] = 12
    mpl.rcParams['font.size'] = 12
    
    # find the frames
    print('> collecting frames')
    frames = glob.glob(_ARGS._FRAME)
    if len(frames) == 0:
        print('ERROR: Not enough frames!')
        raise SystemExit
    stem, _, _ = get_frame_info(os.path.basename(frames[0]))
    
    # setup the logger and the out file
    level    = logging.INFO
    format   = '%(message)s'
    handlers = [logging.FileHandler('Peakhunt_{}.log'.format(stem), 'w'), logging.StreamHandler()]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    out_name = 'Peakhunt_{}.csv'.format(stem)
    
    logging.info('> reading .raw files')
    raw_data = []
    if _ARGS._RAW is not None:
        for raw in glob.glob(_ARGS._RAW):
            raw_data.append(np.rint(np.genfromtxt(raw, usecols=(0,1,2,3,13,14,15), delimiter=[4,4,4,8,8,4,8,8,8,8,8,8,3,7,7,8,7,7,8,6,5,7,7,7,2,5,9,7,7,4,6,11,3,6,8,8,8,8,4])).astype(int))
    
    logging.info('> checking directories')
    if _ARGS._OUTPATH is None:
        out_path = os.getcwd()
    else:
        out_path = os.path.abspath(_ARGS._OUTPATH)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    
    logging.info('> gathering frame information')
    # calculate some frame specific
    # constants once in the beginning
    temp_header, temp_data = read_sfrm(frames[0])
    # read the first frame
    # get image dimensions
    rows, cols = temp_data.shape
    # calculate reduced image dimension (scale of 512x512)
    # to find coordinates in the .raw files
    red_dim = max(temp_data.shape) / 512
    #'CUMULAT': Accumulated exposure time in real hours
    #'ELAPSDR': Requested time for this frame in seconds
    #'ELAPSDA': Actual time for this frame in seconds
    fexp = float(re.findall('\s*ELAPSDA\s*:\s*(\d+\.\d+)', temp_header)[0])
    # Magnitude of scan range in decimal degrees
    fwth = float(re.findall('\s*RANGE\s*:\s*(\d+\.\d+)', temp_header)[0])
    # full queue size
    q = 2 * _ARGS._QUEUE + 1
    # grid stepsize
    s = _ARGS._QUEUE * float(fwth)
    # stepping grid in frames
    xgrid_f = np.linspace(-s, s, q)
    # v = rotation speed: deg/sec
    v = fwth / fexp
    # stepping grid in seconds
    xgrid_s = np.linspace(xgrid_f[0]/v, xgrid_f[-1]/v, len(xgrid_f)*10)
    # stepsize, obsolete but used by the fitting functions
    dx = fwth
    logging.info(' - exposure : {:6.2f}s\n - scanwidth: {:6.2f}deg'.format(fexp, fwth))
    
    # find maxima
    frame_data, local_max = find_peaks(frames, _ARGS._QUEUE, _ARGS._MININT)
    if len(local_max) == 0:
        logging.info('ERROR: Not enough data!')
        raise SystemExit
    
    # filter peaklist
    filtered = filter_peaks(local_max, _ARGS._QUEUE)
    if len(filtered) == 0:
        logging.info('ERROR: Not enough peaks!')
        raise SystemExit
    
    cryst_mosaic = []
    with open(os.path.join(out_path, out_name), 'w') as wfile, PdfPages(os.path.join(out_path, '{}-fit.pdf'.format(stem))) as pdf:
        wfile.write('Gauss;Sum;{};HKL;Frame;Profile\n'.format(';'.join(['Iraw_{:02}'.format(i) for i,v in enumerate(raw_data)])))
        for (name, idx, px, py, peak) in filtered:
            logging.info('\n| >>> Peak @ {:>4}x{:<4} on {}'.format(py, rows - px -1, os.path.basename(name)))
            # procure data from list
            f1 = frame_data[idx]
            # track peak profile
            # add intensities to list
            prof_list = []
            for i in range(idx - _ARGS._QUEUE, idx + _ARGS._QUEUE + 1):
                if len(frames) <= i:
                    logging.info('ERROR: Please increase number of frames')
                    raise SystemExit
                f2 = frame_data[i]
                if _ARGS._USEMAT:
                    ms = 1
                    pInt = np.max(f2[px-ms:px+1+ms, py-ms:py+1+ms])
                else:
                    pInt = f2[px, py]
                prof_list.append(pInt)
                logging.info('|  Profile > {:7}'.format(pInt))
            # peak profile data
            prof_data = np.asarray(prof_list)
            
            # Mosaicity
            I0 = np.sum(prof_data)
            p0 = [I0, 0, 0.2]
            m_popt, m_pcov = curve_fit(f_gauss_ang, xgrid_f, prof_data, p0=p0)
            mosaic = float(m_popt[2])
            cryst_mosaic.append(mosaic)
            
            # FIT PEAKS GAUSS/LORENTZ
            # initial guesses
            # sigma or gamma (gauss / lorentz)
            I0 = np.sum(prof_data)
            p0 = [I0, 0, 0.2]
            g_popt, g_pcov = curve_fit(I_gauss, xgrid_f, prof_data, p0=p0)
            g_err = np.sqrt(np.diag(g_pcov))
            g_max = g_popt[0]/((g_popt[2]/v)*np.sqrt(2*np.pi))
            logging.info('|\n| >>> Gaussian Fitted Peak')
            logging.info('|  Integrated pixel intensity (sum) [phts]: {:.2e}'.format(I0))
            logging.info('|  Integrated pixel intensity (fit) [phts]: {:.2e}'.format(g_popt[0]))
            logging.info('|  Mosaicity (fit) [deg]: {:.2f}'.format(mosaic))
            logging.info('|  FWHM of count rate function [sec]: {:.2f}'.format(g_popt[2]*2.355))
            logging.info('|  Maximum count rate [phts/s]: {:.2e}'.format(g_max))
            
            if _ARGS._TAILS:
                idx_omit = np.argmax(prof_data)
                o_xgrid_f = np.delete(xgrid_f, idx_omit)
                o_prof_data = np.delete(prof_data, idx_omit)
                t_popt, s_pcov = curve_fit(I_gauss, o_xgrid_f, o_prof_data, p0=p0)
                t_err = np.sqrt(np.diag(s_pcov))
                t_max = t_popt[0]/((t_popt[2]/v)*np.sqrt(2*np.pi))
                logging.info('|\n| >>> Gaussian Shoulder Fit')
                logging.info('|  Integrated pixel intensity (fit) [phts]: {:.2e}'.format(t_popt[0]))
                logging.info('|  FWHM of count rate function [sec]: {:.2f}'.format(t_popt[2]*2.355))
                logging.info('|  Maximum count rate [phts/s]: {:.2e}'.format(t_max))
            
            if _ARGS._LORENTZ:
                l_popt, l_pcov = curve_fit(I_lorentz, xgrid_f, prof_data, p0=p0)
                l_max = l_popt[0]/(np.pi*l_popt[2]/v)
                logging.info('|\n| >>> Lorentzian Fitted Peak')
                logging.info('|  Integrated pixel intensity (fit) [phts]: {:.2e}'.format(l_popt[0]))
                logging.info('|  FWHM of count rate function [sec]: {:.2f}'.format(l_popt[2]*2))
                logging.info('|  Maximum count rate [phts/s]: {:.2e}'.format(l_max))
            
            if _ARGS._RAW is not None:
                # IDENTIFY HKL > FIND HKL IN RAW
                # XO F7.2 Observed X-pixel coordinate of the
                #         intensity-weighted reflection centroid,
                #         in reduced pixels (scale of 512x512)
                # 
                # YO F7.2 Observed Y-pixel coordinate of the
                #         intensity-weighted reflection centroid,
                #         in reduced pixels (scale of 512x512)
                # 
                # ZO F8.2 Observed frame number of the
                #         intensity-weighted reflection centroid
                # 
                # new x, y: maximum image dimension / 512
                # adjust peak position to Bruker convention
                # to find the peak in APEX: 
                # - switch x and y or rows and cols
                # - flip rows: (rows - px -1)
                bruker_py = int(round((rows - px -1) / red_dim, 0))
                bruker_px = int(round(py / red_dim, 0))
                raw_int = []
                fnam = os.path.basename(frames[idx])
                for raw in raw_data:
                    _, rnum, fnum = get_frame_info(fnam)
                    x = np.asarray([bruker_px, bruker_py, fnum]).astype(int)
                    vals = raw[(raw[:,4:] >= x-1).all(1) & (raw[:,4:] <= x+1).all(1)]
                    h,k,l = 0,0,0
                    if len(vals) > 0:
                        h,k,l = vals[0][:3]
                        raw_int.append(vals[0][3])
                
                logging.info('|\n| >>> Integrated Intensity')
                logging.info('|  Peak in .raw @ {:>4}x{:<4}'.format(bruker_px, bruker_py))
                
                if len(raw_int) == 1:
                    raw_int = round(raw_int[0], 0)
                    raw_int_s = '{:7}'.format(raw_int)
                elif len(raw_int) > 1:
                    logging.info('|  inconclusive')
                    raw_int_s = 'inconclusive'
                    continue
                else:
                    logging.info('|  untraceable')
                    raw_int_s = 'untraceable'
                    continue
                
                logging.info('|  int: {} HKL: ({:3}{:3}{:3}) Frame: {}'.format(raw_int_s, h, k, l, fnam))
                wfile.write('{};{};{};{:3}{:3}{:3};{};{}\n'.format(int(g_max), I0, raw_int_s, h, k, l, fnam, ';'.join(map(str, f_gauss(xgrid_s, *g_popt)))))
            
            if _ARGS._PLOT:
                fig, (p11,p12) = plt.subplots(1,2)
                p11.set_title('[cnts]')
                p12.set_title('[phts/s]')
                p11.set_xlabel('Degree')
                p11.set_ylabel('Counts / Pixel')
                p12.set_xlabel('Seconds')
                p12.set_ylabel('Photons / Second')
                p11.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                p12.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                data = p11.plot(xgrid_f, prof_data, 'k*', label='Data')
                gf = p11.plot(xgrid_f, I_gauss(xgrid_f, *g_popt), '-', color='#37a0cb', label='G: {:.2e}'.format(g_popt[0]))
                gs = p12.plot(xgrid_s, f_gauss(xgrid_s, *g_popt), '-', color='#37a0cb', label='G: {:.2e}'.format(g_max))
                fig.suptitle('Integrated pixel intensity (sum) [phts]: {:.2e}\nMosaicity (fit) [deg]: {:.2f}'.format(I0, mosaic))
                plt.annotate('G:Gaussian fit, T:Gaussian fit tails only, L:Lorentzian fit', xy=(0.5, 0.02), xycoords='figure fraction', ha='center')
                if _ARGS._RAW is not None:
                    plt.annotate('raw [ctns]: {}'.format(raw_int_s), xy=(0.5, 0.92), xycoords='figure fraction', ha='center')
                if _ARGS._TAILS:
                    sf = p11.plot(o_xgrid_f, I_gauss(o_xgrid_f, *t_popt), '-', color='#00aba4', label='T: {:.2e}'.format(t_popt[0]))
                    ss = p12.plot(xgrid_s, f_gauss(xgrid_s, *t_popt), '-', color='#00aba4', label='T: {:.2e}'.format(t_max))
                if _ARGS._LORENTZ:
                    lf = p11.plot(xgrid_f,I_lorentz(xgrid_f, *l_popt), '-', color='#003d73', label='L: {:.2e}'.format(l_popt[0]))
                    ls = p12.plot(xgrid_s,f_lorentz(xgrid_s, *l_popt), '-', color='#003d73', label='L: {:.2e}'.format(l_max))
                p11.legend()
                p12.legend()
                pdf.savefig()
                plt.close()
    
    cryst_mosaic = np.asarray(cryst_mosaic)
    logging.info('\nEst. mosaicity: {:.2f} ({:.2f} - {:.2f})'.format(np.mean(cryst_mosaic), np.min(cryst_mosaic), np.max(cryst_mosaic)))

if __name__ == '__main__':
    main()