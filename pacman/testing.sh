#!/bin/bash

#{1..2} {5..7} {9..19} "21" {23..25}
NUM_RUNS=50
PACMAN_LIST=("23120049")
GHOST_LIST=("25")
LOG_FILE="game_results.log"

echo "START GAME $NUM_RUNS"
echo "Pacman: $PACMAN"
echo "Ghost: $GHOST"

for PACMAN in "${PACMAN_LIST[@]}"; do
    
    for GHOST in "${GHOST_LIST[@]}"; do
        
        echo "MATCH: Pacman ($PACMAN) vs Ghost ($GHOST)"

        > "$LOG_FILE"
        for i in $(seq 1 $NUM_RUNS); do
            echo -n "." 
        
            bash run_game.sh "$PACMAN" "$GHOST" | grep -E "^(pacman_wins|ghost_wins|draw)$" >> "$LOG_FILE"
        done
        echo "" 
        echo "END MATCH: $PACMAN vs $GHOST"

        python3 analyze_result.py "$LOG_FILE" "$PACMAN" "$GHOST"
        > "$LOG_FILE"

    done
done

