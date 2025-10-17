from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict


router = APIRouter()


class SpeciesItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    species: str
    scientific_name: str = Field(alias="scientificName")


@router.get("/species/search", response_model=list[SpeciesItem])
async def search_species(q: str):
    q_lower = q.lower()
    all_species = [
        ("Green Anole", "Anolis carolinensis"),
        ("Brown Anole", "Anolis sagrei"),
        ("Crested Anole", "Anolis cristatellus"),
        ("Knight Anole", "Anolis equestris"),
        ("Bark Anole", "Anolis distichus"),
    ]
    results: list[SpeciesItem] = []
    for common, scientific in all_species:
        if q_lower in common.lower() or q_lower in scientific.lower():
            results.append(SpeciesItem(species=common, scientific_name=scientific))
    return results


class SpeciesDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    species: str
    scientific_name: str = Field(alias="scientificName")
    description: str
    habitat: str


@router.get("/species/{scientific_name}", response_model=SpeciesDetails)
async def get_species_details(scientific_name: str):
    details = {
        "Anolis carolinensis": SpeciesDetails(
            species="Green Anole",
            scientific_name="Anolis carolinensis",
            description="A small, arboreal anole native to the southeastern United States.",
            habitat="Trees, shrubs, and suburban areas",
        ),
        "Anolis sagrei": SpeciesDetails(
            species="Brown Anole",
            scientific_name="Anolis sagrei",
            description="An invasive species common in Florida; often found on the ground and low vegetation.",
            habitat="Ground cover, shrubs, urban landscapes",
        ),
        "Anolis cristatellus": SpeciesDetails(
            species="Crested Anole",
            scientific_name="Anolis cristatellus",
            description="A species native to Puerto Rico, introduced to Florida.",
            habitat="Urban and suburban habitats",
        ),
        "Anolis equestris": SpeciesDetails(
            species="Knight Anole",
            scientific_name="Anolis equestris",
            description="A large anole species native to Cuba; introduced to Florida.",
            habitat="Trees in urban and suburban habitats",
        ),
        "Anolis distichus": SpeciesDetails(
            species="Bark Anole",
            scientific_name="Anolis distichus",
            description="Often seen on tree trunks; introduced to Florida.",
            habitat="Tree trunks, urban habitats",
        ),
    }
    return details.get(
        scientific_name,
        SpeciesDetails(
            species=scientific_name,
            scientific_name=scientific_name,
            description="Details not found.",
            habitat="Unknown",
        ),
    )
