import csv
import random
import os
from datetime import datetime, timedelta

# Categories and their proportions
labels = {
    "Accident": 1500,
    "Incident": 1500,
    "Near Miss": 1000,
    "Safety Observation": 1000
}

# Template phrases for each category
templates = {
    "Accident": [
        "Worker fell from {height} while {task}, resulting in {injury}.",
        "{equipment} malfunction caused {damage}, injuring {victim}.",
        "Exposure to {hazard} led to {injury}."
    ],
    "Incident": [
        "{equipment} failure caused {damage} to {asset}.",
        "{substance} spill contaminated {area}.",
        "Structural collapse in {location} due to {cause}."
    ],
    "Near Miss": [
        "Worker nearly {action} but avoided injury.",
        "Unsecured {object} almost fell from {height}.",
        "{hazard} was discovered before harm occurred."
    ],
    "Safety Observation": [
        "All {equipment} was used correctly during {task}.",
        "{team} followed {protocol} without violations.",
        "Negative: {condition} was observed but not corrected."
    ]
}

# Fillers for templates
fillers = {
    "height": ["scaffolding", "ladder", "ship deck"],
    "task": ["welding", "painting", "repairing the hull"],
    "injury": ["a fractured leg", "burns", "concussion"],
    "equipment": ["crane", "forklift", "grinder"],
    "damage": ["a steel beam to drop", "a fuel leak", "electrical fire"],
    "victim": ["the operator", "two workers", "a contractor"],
    "hazard": ["toxic fumes", "asbestos", "high-voltage wiring"],
    "asset": ["dry dock floor", "ship propeller", "storage area"],
    "substance": ["Oil", "Chemicals", "Hydraulic fluid"],
    "area": ["dock area", "engine room", "paint shop"],
    "location": ["dry dock", "shipyard warehouse", "pier"],
    "cause": ["overload", "corrosion", "improper maintenance"],
    "action": ["slipped on an oily surface", "touched live wiring", "stepped into an open hatch"],
    "object": ["ladder", "toolbox", "pipe"],
    "condition": ["untidy cables", "missing guardrails", "blocked emergency exit"],
    "team": ["Paint crew", "Welding team", "Scaffolders"],
    "protocol": ["lockout/tagout", "PPE guidelines", "confined space entry"]
}

# Name, Location, and Department pools
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

# Expanded injury descriptions for realistic workplace incidents
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

# Generate random incident date (from Jan 1, 2024 to today)
def random_date():
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    incident_date = start_date + timedelta(days=random_days)
    return incident_date.strftime("%Y-%m-%d")

# Generate random incident time
def random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

# Generate incident description
def generate_entry(label):
    template = random.choice(templates[label])
    entry = template.format(**{k: random.choice(v) for k, v in fillers.items() if k in template})
    return entry

# Ensure 'Training Data' folder exists
output_folder = "Training Data"
os.makedirs(output_folder, exist_ok=True)

# Output file path
output_file = os.path.join(output_folder, 'shipyard_structured_dataset.csv')

# Generate CSV
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'reporter_name', 'person_involved', 'incident_date', 'incident_time',
        'department', 'incident_description', 'location', 'label',
        'was_injured', 'injury_description'
    ])
    
    for label, count in labels.items():
        for _ in range(count):
            description = generate_entry(label)
            reporter_name = f"{random.choice(first_names)} {random.choice(surnames)}"
            person_involved = f"{random.choice(first_names)} {random.choice(surnames)}"
            incident_date = random_date()
            incident_time = random_time()
            location = random.choice(locations)
            department = random.choice(departments)
            
            if label in ["Accident", "Incident"]:
                was_injured = "Yes"
                injury_desc = random.choice(injury_descriptions)
            else:
                was_injured = "No"
                injury_desc = "N/A"
            
            writer.writerow([
                reporter_name, person_involved, incident_date, incident_time,
                department, description, location, label,
                was_injured, injury_desc
            ])
