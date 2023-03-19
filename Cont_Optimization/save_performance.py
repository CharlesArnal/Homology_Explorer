import os
 
 
def save_performance(best_score,time_passed,period,perf_archive_file):
    """
    Will be slow if the file gets very large (more than thousands of lines), fine otherwise
    Saves scores in the following format on each line: time_since_beginning score
    """
    with open(perf_archive_file, 'a+') as f:
        f.seek(0)
        if f.readlines() == []:
            f.write(f"{time_passed} {best_score}")
        else:
            f.seek(0)
            last_line = f.readlines()[-1]
            previous_time = float(last_line.split(" ")[0])
            if time_passed-previous_time>= period:
                f.write(f"\n{time_passed} {best_score}")
        
   
    
#save_performance(2,120,20, r'C:\Users\CombinatorialRL\Code\test_saving_perf.txt')
    