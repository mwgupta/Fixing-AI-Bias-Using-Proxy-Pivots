import csv

#with open('adult.txt') as csv_file:
csv_reader = csv.reader(compas-scores-raw.csv, delimiter=',')
male = 0
female = 0

count = 1
for row in csv_reader:
    print(count)
    count += 1
        """
        if count == 32561 or count == 0:
            break
        count +=1 
        race = row[9]
        if race == " Male":
            male += 1
        else:
            female += 1
            
    print("male: " + str(male))
    print("female: " + str(female))
    """
  
