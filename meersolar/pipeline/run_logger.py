from meersolar.pipeline.basic_func import get_datadir

def run_logger(logdir):
    datadir=get_datadir()
    logger=datadir+"/MeerLogger.AppImage "
    os.system(f"{logger} {logdir}")
    
def main():
    usage = "MeerSOLAR Logger"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--job_id",
        dest="job_id",
        default=None,
        help="MeerSOLAR Job ID",
        metavar="Integer",
    )
    parser.add_option(
        "--logdir",
        dest="logdir",
        default=None,
        help="Name of log directory",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.job_id==None and options.logdir==None:
        print ("Please provide either job ID or log directory.")
        return
    else:   
        datadir=get_datadir()
        if options.job_id!=None:
            jobfile_name=datadir + f"/main_pids_{options.job_id}.txt"
            if os.path.exists(jobfile_name)==False:
                print (f"Job ID: {options.job_id} is not available. Provide log directory name.")
                return 1
            else:
                results=np.loadtxt(jobfile_name,dtype="str",unpack=True)
                basedir=results[2]
                if os.path.exists(basedir)==False:
                    print ("Base directory : {basedir} is not present.")
                    return 1 
                logdir=basedir+"/logs"
        else:
            if os.path.exists(options.logdir)==False:
                print (f"Log diretory: {options.logdir} is not present. Please provide a valid log directory.")
                return 1
            logdir=options.logdir
    try:
        run_logger(logdir)
    except Exception as e:
        traceback.print_exc()
        return msg
    
if __name__ == "__main__":
    result = main()
    os._exit(result)
    

