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
    "Office - Admin Block", "Warehouse A", "Warehouse B", "Dockyard North", "Dockyard South",
    "Dry Dock 1", "Dry Dock 2", "Shipyard Pier", "Engineering Workshop", "IT Server Room", 
    "Main Gate Security", "Cafeteria", "Maintenance Shed", "Training Room", "Fleet Garage", 
    "Reception Area", "Tool Crib", "Paint Shop", "Electrical Bay", "Fuel Depot"
]

departments = [
    "Warehouse", "Customer Service", "IT", "Engineering", "Operations", "Procurement",
    "Health & Safety", "Dockyard Operations", "Ship Maintenance", "Logistics", "HR",  
    "Security", "Training & Development", "Fleet Management", "Facilities", 
    "Environmental Services", "Quality Assurance", "Finance", "Legal", 
    "Communications", "Research & Development"
]

injury_descriptions = [
    "Laceration from sharp metal panel",
    "Crush injury to hand from hydraulic press",
    "Burn from hot welding equipment",
    "Fractured ankle due to fall from scaffolding",
    "Chemical burn from solvent exposure",
    "Electric shock from faulty wiring",
    "Concussion from falling object",
    "Back strain from improper lifting",
    "Dislocated shoulder during equipment maintenance",
    "Hearing damage from loud machinery",
    "Eye irritation from chemical fumes",
    "Minor cuts and bruises",
    "Severe abrasion from slip on deck",
    "Fractured wrist from fall on wet surface",
    "Puncture wound from misplaced tool",
    "Heat exhaustion from prolonged outdoor work",
    "Smoke inhalation from fire in electrical bay",
    "Foreign object in eye from grinding debris",
    "Trip hazard fall resulting in bruising",
    "Compressed air injury during maintenance"
]

incident_descriptions = [
    "Worker fell from scaffolding during welding, resulting in burns.",
    "Forklift malfunction caused a fuel leak, injuring the operator.",
    "Exposure to toxic fumes led to a fractured leg.",
    "Crane failure caused electrical fire to dry dock floor.",
    "Oil spill contaminated dock area.",
    "Structural collapse in dry dock due to corrosion.",
    "Worker nearly slipped on an oily surface but avoided injury.",
    "Unsecured ladder almost fell from scaffolding.",
    "Toxic fumes were discovered before harm occurred.",
    "All equipment was used correctly during painting.",
    "Paint crew followed PPE guidelines without violations.",
    "Negative: untidy cables were observed but not corrected."
]

labels = ["Accident", "Incident", "Near Miss", "Safety Observation"]

# Random date generator
def random_date():
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

# Random time generator
def random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

# Generate merged unstructured text
def generate_full_text(fields):
    keys = list(fields.keys())
    random.shuffle(keys)
    text = " ".join([f"{k.replace('_', ' ').title()}: {fields[k]}" for k in keys])

    return text

# Ensure output folder
output_folder = "Training Data"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "shipyard_unstructure_with_structure_dataset.csv")

# Write CSV with only full_text
with open(output_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["full_text"])
    
    for _ in range(5000):  # Generate 5000 rows
        label = random.choice(labels)
        was_injured = "Yes" if label in ["Accident", "Incident"] else "No"
        injury_description = random.choice(injury_descriptions) if was_injured == "Yes" else "N/A"
        
        fields = {
            "reporter_name": f"{random.choice(first_names)} {random.choice(surnames)}",
            "person_involved": f"{random.choice(first_names)} {random.choice(surnames)}",
            "incident_date": random_date(),
            "incident_time": random_time(),
            "department": random.choice(departments),
            "incident_description": random.choice(incident_descriptions),
            "location": random.choice(locations),
            "label": label,
            "was_injured": was_injured,
            "injury_description": injury_description
        }
        
        full_text = generate_full_text(fields)
        
        writer.writerow([full_text])
