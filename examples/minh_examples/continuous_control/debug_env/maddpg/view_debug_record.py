
import glob
import os 
from datetime import datetime

def main():


    mava_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") # change to actual name

    # Recordings
    list_of_files = glob.glob(f"/root/mava/{mava_id}/recordings/*.html")

    if(list_of_files == 0):
        print("No recordings are available yet. Please wait or run the 'Run Multi-Agent DDPG System.' cell if you haven't already done this.")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        print("Run the next cell to visualize your agents!")

    

if __name__ == "__main__":
    main() 