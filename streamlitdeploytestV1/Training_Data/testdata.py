import csv
import random
from datetime import datetime, timedelta

# Sample data pools
locations = [
    "Paint Shop", "Electrical Bay", "Fleet Garage", "Main Gate Security", "Tool Crib",
    "Office - Admin Block", "Maintenance Shed", "Dry Dock 1", "Dry Dock 2",
    "Reception Area", "Warehouse B", "Shipyard Pier", "Welding Station"
]

departments = [
    "Health & Safety", "Dockyard Operations", "Environmental Services", "Legal",
    "Communications", "Finance", "IT", "Quality Assurance", "Fleet Management",
    "Customer Service", "Engineering", "Warehouse"
]

injuries = [
    "Dislocated shoulder during equipment maintenance",
    "Exposure to high-voltage wiring led to concussion",
    "Hearing damage from loud machinery",
    "Exposure to toxic fumes led to burns",
    "Minor cuts and bruises",
    "Fractured wrist from fall on wet surface",
    "Grinder malfunction caused a steel beam to drop, injuring a contractor",
    "Crane malfunction caused a steel beam to drop, injuring a contractor",
    "Puncture wound from misplaced tool",
    "Compressed air injury during maintenance",
    "Electric shock from faulty wiring",
    "Worker fell from ladder while welding, resulting in burns",
    "Forklift malfunction caused electrical fire, injuring a contractor",
    "Chemical burn from solvent exposure",
    "Trip hazard fall resulting in bruising",
    "Eye irritation from chemical fumes",
    "Heat exhaustion from prolonged outdoor work",
    "Smoke inhalation from fire in electrical bay",
    "Worker fell from scaffolding while painting, resulting in burns",
    "Fractured ankle due to fall from scaffolding",
    "Grinder malfunction caused electrical fire, injuring two workers",
    "Forklift malfunction caused a fuel leak, injuring the operator"
]

first_names = ["Avery", "Kai", "Jordan", "Skyler", "Peyton", "Frankie", "Riley", "Sydney", "Quinn", "Harper", "Jesse", "Rowan", "Charlie", "Spencer", "Morgan", "Reese", "Dakota"]
last_names = ["Smith", "Johnson", "Brown", "Taylor", "Clark", "Lewis", "Walker", "Robinson", "Lee", "Foster", "Mitchell", "Payne", "Kennedy", "Holmes", "Reed", "Pearson", "Bennett"]

# Generate random datetime
def random_datetime(start_year=2024, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86400)
    dt = start + timedelta(days=random_days, seconds=random_seconds)
    return dt.strftime("On %d %B %Y at %H:%M")

# Generate one incident report
def generate_incident():
    timestamp = random_datetime()
    location = random.choice(locations)
    injury1 = random.choice(injuries)
    injury2 = random.choice(injuries)
    reporter = f"{random.choice(first_names)} {random.choice(last_names)}"
    involved = f"{random.choice(first_names)} {random.choice(last_names)}"
    department = random.choice(departments)
    return f"{timestamp}, an incident occurred at {location}. {injury1} {injury2}. {involved} from the {department} department was involved. The incident was reported by {reporter}."

# Write to CSV
with open("test_incident_reports.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text"])
    for _ in range(100):
        writer.writerow([generate_incident()])

print("✅ incident_reports.csv has been created with 100 rows.")
