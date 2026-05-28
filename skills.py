"""Collect team member skills from terminal and save to JSON."""

import json


def parse_skills(raw_skills: str) -> list[str]:
    """Convert comma-separated skills into a cleaned list."""
    return [skill.strip() for skill in raw_skills.split(",") if skill.strip()]


def main() -> None:
    profiles: dict[str, list[str]] = {}

    print("Enter team members. Type 'done' as the name to finish.")

    while True:
        name = input("Member name: ").strip()

        if name.lower() == "done":
            break

        if not name:
            print("Name cannot be empty. Try again.")
            continue

        skills_input = input("Skills (comma separated): ").strip()
        profiles[name] = parse_skills(skills_input)

    with open("skills.json", "w", encoding="utf-8") as file:
        json.dump(profiles, file, indent=2, ensure_ascii=False)

    print("\nFinal profiles:")
    print(json.dumps(profiles, indent=2, ensure_ascii=False))
    print("\nSaved to skills.json")


if __name__ == "__main__":
    main()
