import sys
VALID_RESULTS = ["pacman_wins", "ghost_wins", "draw"]

def analyze_n_save(log_file, pac_name, ghost_name):

    result_dict = {
        "pacman_wins" : 0,
        "ghost_wins" : 0,
    }

    total_matches = 0

    with open(log_file, "r") as file:
        for line in file:
            line = line.strip()
            
            if line == "pacman_wins":
                result_dict[line] += 1
                total_matches += 1
            elif line == "ghost_wins":
                result_dict[line] += 1
                total_matches +=1
                
    pacman_rate = round((result_dict['pacman_wins']*100/total_matches), 2)
    ghost_rate = round((result_dict['ghost_wins']*100/total_matches), 2)
                
    print(f"Pacman win: {pacman_rate}%")
    print(f"Ghost win: {ghost_rate}%")
    
    with open("result_summary.txt", "a") as file:
        file.write(f"Pacman: {pac_name} || Ghost: {ghost_name} || Pacman rate: {pacman_rate} || Ghost rate: {ghost_rate}\n")
        
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("ERROR")
        sys.exit(1)

    log_file_path = sys.argv[1]
    pacman_agent_name = sys.argv[2]
    ghost_agent_name = sys.argv[3]
    
    analyze_n_save(log_file_path, pacman_agent_name, ghost_agent_name)