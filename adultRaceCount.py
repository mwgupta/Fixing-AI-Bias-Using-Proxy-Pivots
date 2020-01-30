import csv

with open('adult.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    white = 0
    black = 0
    asian_pac = 0
    amer_ind = 0
    other = 0

    count = 0
    for row in csv_reader:
        if count == 32561:
            break
        count +=1 
        race = row[8]
        if race == " White":
            white += 1
        elif race == " Black":
            black += 1
        elif race == " Asian-Pac-Islander":
            asian_pac += 1
        elif race == " Amer-Indian-Eskimo":
            amer_ind += 1
        else:
            other += 1
            
    print("white: " + str(white))
    print("black: " + str(black))
    print("asian-pac-islander: " + str(asian_pac))
    print("amer-indian-eskimo: " + str(amer_ind))
    print("other: " + str(other))
