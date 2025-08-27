import csv
import random
import os
from datetime import datetime, timedelta

# Data pools
first_names = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Blake", "Jamie", "Chris", "Drew", "Cameron",
    "Riley", "Sydney", "Quinn", "Avery", "Dakota", "Reese", "Skyler", "Peyton", "Emerson", "Finley",
    "Logan", "Harper", "Elliot", "Charlie", "Frankie", "Sawyer", "Rowan", "Kai", "Jesse", "Spencer"
]

surnames = [
    "Smith", "Johnson", "Lee", "Walker", "Brown", "Davis", "Clark", "Lewis", "Allen", "Robinson",
    "Mitchell", "Parker", "Murphy", "Bailey", "Cooper", "Reed", "Foster", "Graham", "Hughes", "Ward",
    "Sullivan", "Morgan", "Murray", "Fisher", "Payne", "Kennedy", "Bennett", "Pearson", "Holmes", "West"
]

locations = [
    "the Admin Block", "Warehouse A", "Warehouse B", "Dockyard North", "Dockyard South",
    "Dry Dock 1", "Dry Dock 2", "the Shipyard Pier", "the Engineering Workshop", "the IT Server Room", 
    "Main Gate Security", "the Cafeteria", "the Maintenance Shed", "the Training Room", "Fleet Garage", 
    "Reception Area", "the Tool Crib", "the Paint Shop", "the Electrical Bay", "the Fuel Depot"
]

departments = [
    "Warehouse", "Customer Service", "IT", "Engineering", "Operations", "Procurement",
    "Health and Safety", "Dockyard Operations", "Ship Maintenance", "Logistics", "HR",  
    "Security", "Training and Development", "Fleet Management", "Facilities", 
    "Environmental Services", "Quality Assurance", "Finance", "Legal", 
    "Communications", "Research and Development"
]

injury_descriptions = [
    "suffered a laceration from a sharp metal panel.",
    "sustained a crush injury to the hand while operating machinery.",
    "received burns from welding equipment.",
    "fractured an ankle due to a fall from scaffolding.",
    "suffered a chemical burn from solvent exposure.",
    "experienced an electric shock while handling faulty wiring.",
    "was diagnosed with a concussion after being struck by a falling object.",
    "strained their back while lifting heavy equipment.",
    "dislocated a shoulder during maintenance activities.",
    "suffered hearing damage due to prolonged noise exposure.",
    "had eye irritation caused by chemical fumes.",
    "received minor cuts and bruises.",
    "suffered severe abrasions after slipping on deck.",
    "fractured a wrist due to a fall on a wet surface.",
    "suffered a puncture wound from a misplaced tool.",
    "experienced heat exhaustion from outdoor work.",
    "inhaled smoke during a fire incident.",
    "had a foreign object lodged in their eye from debris.",
    "fell due to a trip hazard, resulting in bruising.",
    "sustained a compressed air injury during maintenance."
]

incident_descriptions = [
    "A worker fell from scaffolding during welding operations, resulting in injury.",
    "A forklift malfunction caused a fuel leak, leading to injury.",
    "Toxic fumes exposure resulted in an injury.",
    "A crane failure caused an electrical fire on the dry dock floor.",
    "An oil spill contaminated the dock area.",
    "Structural collapse occurred in the dry dock due to corrosion.",
    "A worker nearly slipped on an oily surface but avoided injury.",
    "An unsecured ladder almost fell from scaffolding.",
    "Toxic fumes were detected before any harm occurred.",
    "All equipment was used correctly during painting tasks.",
    "The paint crew followed all PPE guidelines without violations.",
    "Negative observation: untidy cables were noted but not corrected."
]

labels = ["Accident", "Incident", "Near Miss", "Safety Observation"]

# Random date generator
def random_date():
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).strftime("%d %B %Y")

# Random time generator
def random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

# Generate natural-sounding incident text with no explicit labels
def generate_natural_text():
    reporter = f"{random.choice(first_names)} {random.choice(surnames)}"
    person = f"{random.choice(first_names)} {random.choice(surnames)}"
    date = random_date()
    time = random_time()
    dept = random.choice(departments)
    incident = random.choice(incident_descriptions)
    loc = random.choice(locations)
    label = random.choice(labels)
    was_injured = "Yes" if label in ["Accident", "Incident"] else "No"
    injury = random.choice(injury_descriptions) if was_injured == "Yes" else ""

    text_parts = [
        f"On {date} at {time}, an incident occurred at {loc}.",
        f"{person} from the {dept} department was involved.",
        incident,
        injury,
        f"The incident was reported by {reporter}."
    ]
    random.shuffle(text_parts)
    return " ".join([part for part in text_parts if part.strip()])

# Ensure output folder
output_folder = "Training Data"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "shipyard_unstructured_without_structure_dataset.csv")

# Write CSV with only unstructured text
with open(output_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["full_text"])
    
    for _ in range(5000):
        full_text = generate_natural_text()
        writer.writerow([full_text])
