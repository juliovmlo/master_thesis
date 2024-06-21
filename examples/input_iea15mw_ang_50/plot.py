import matplotlib.pyplot as plt
import pandas as pd

# Define the column names based on the provided data
columns = ["Radius_s", "alfa", "Vrel", "Cl", "Cd", "Cm", "L", "D", "Pos_RP x", "Pos_RP y", "Pos_RP z", 
           "Secfrc_RPx", "Secfrc_RPy", "Secfrc_RPz", "Secmom_RPx", "Secmom_RPy", "Secmom_RPz", 
           "Intmom_RPx", "Intmom_RPy", "Intmom_RPz"]

# Load the data from the file
file_path ='res/output_at_200s.dat' 
data = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=columns)

# Plot Pos_RP y against Radius_s
plt.figure(figsize=(10, 6))
plt.plot(data["Radius_s"], data["Pos_RP y"], marker='o', linestyle='-')
plt.plot(data["Radius_s"], data["Pos_RP x"], marker='o', linestyle='-')
plt.xlabel("Radius_s")
plt.ylabel("Pos_RP y")
plt.title("Pos_RP y vs. Radius_s")
plt.grid(True)
plt.show()