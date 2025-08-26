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

# Generate incident description
def generate_entry(label):
    template = random.choice(templates[label])
    entry = template.format(**{k: random.choice(v) for k, v in fillers.items() if k in template})
    return entry

# Generate natural-sounding incident text with no explicit labels
def generate_natural_text(fields):
    # Use the actual row's fields to build the narrative
    text_parts = [
        f"On {fields['incident_date']} at {fields['incident_time']}, an incident occurred at {fields['location']}.",
        f"{fields['person_involved']} from the {fields['department']} department was involved.",
        fields['incident_description'],
        fields['injury_description'] if fields['was_injured'] == "Yes" else "",
        f"The incident was reported by {fields['reporter_name']}."
    ]
    random.shuffle(text_parts)
    return " ".join([part for part in text_parts if part.strip()])

# Generate merged unstructured text
def generate_text(fields):
    keys = list(fields.keys())
    random.shuffle(keys)
    text = " ".join([f"{k.replace('_', ' ').title()}: {fields[k]}" for k in keys])

    return text

output_folder = "Training Data"
os.makedirs(output_folder, exist_ok=True)

structured_file = os.path.join(output_folder, 'shipyard_structured_dataset.csv')
unstructured_file = os.path.join(output_folder, 'shipyard_unstructured_without_structure_dataset.csv')
slight_structure_file = os.path.join(output_folder, 'shipyard_unstructure_with_structure_dataset.csv')

with open(structured_file, 'w', newline='') as structured_csv, \
     open(unstructured_file, 'w', newline='') as unstructured_csv, \
     open(slight_structure_file, 'w', newline='') as slight_structure_csv:

    structured_writer = csv.writer(structured_csv)
    unstructured_writer = csv.writer(unstructured_csv)
    slight_structure_writer = csv.writer(slight_structure_csv)

    # Write headers
    structured_writer.writerow([
        'reporter_name', 'person_involved', 'incident_date', 'incident_time',
        'department', 'incident_description', 'location', 'label',
        'was_injured', 'injury_description'
    ])
    unstructured_writer.writerow(['text'])
    slight_structure_writer.writerow(['text'])
    
    for label, count in labels.items():
        for _ in range(count):
            reporter_name = f"{random.choice(first_names)} {random.choice(surnames)}"
            person_involved = f"{random.choice(first_names)} {random.choice(surnames)}"
            incident_date = random_date()
            incident_time = random_time()
            location = random.choice(locations)
            department = random.choice(departments)
            description = generate_entry(label)
            was_injured = "Yes" if label in ["Accident", "Incident"] else "No"
            injury_desc = random.choice(injury_descriptions) if was_injured == "Yes" else "N/A"

            # Structured row
            structured_writer.writerow([
                reporter_name, person_involved, incident_date, incident_time,
                department, description, location, label,
                was_injured, injury_desc
            ])

            # Prepare fields dict for unstructured and slight structure
            fields = {
                'reporter_name': reporter_name,
                'person_involved': person_involved,
                'incident_date': incident_date,
                'incident_time': incident_time,
                'department': department,
                'incident_description': description,
                'location': location,
                'label': label,
                'was_injured': was_injured,
                'injury_description': injury_desc
            }

            # Unstructured text
            unstructured_text = generate_natural_text(fields)
            unstructured_writer.writerow([unstructured_text])

            # Slightly structured text
            slight_structure_text = generate_text(fields)
            slight_structure_writer.writerow([slight_structure_text])
