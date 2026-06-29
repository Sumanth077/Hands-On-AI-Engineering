import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool

load_dotenv()

CROP_CALENDAR_DATA = {
    "tomato": {
        "planting_season": "Start seeds indoors 6-8 weeks before last frost; transplant after soil reaches 60F (15C).",
        "harvest_season": "Harvest 60-85 days after transplant when fruit is firm and fully colored.",
        "notes": "Needs full sun, consistent moisture, and support stakes or cages. Avoid overhead watering to reduce blight risk.",
    },
    "corn": {
        "planting_season": "Direct sow 2 weeks after last frost when soil is at least 60F (15C). Plant in blocks for pollination.",
        "harvest_season": "Pick 18-24 days after silks appear when kernels release milky fluid when pierced.",
        "notes": "Heavy nitrogen feeder. Rotate with legumes. Watch for corn earworm and fall armyworm.",
    },
    "wheat": {
        "planting_season": "Winter wheat: sow in fall 6-8 weeks before first hard frost. Spring wheat: sow as soon as soil can be worked.",
        "harvest_season": "Harvest when grain is hard and moisture is below 20%, typically early to mid summer for spring types.",
        "notes": "Prefers well-drained loam. Scout for rust, aphids, and Hessian fly in winter wheat.",
    },
    "rice": {
        "planting_season": "Transplant seedlings 4-5 weeks after sowing when monsoon rains begin or after flooding paddies in warm regions.",
        "harvest_season": "Harvest 120-150 days after transplant when grains turn golden and moisture drops to 18-22%.",
        "notes": "Requires flooded or saturated fields in many systems. Manage blast, stem borer, and weeds early.",
    },
    "potato": {
        "planting_season": "Plant seed pieces 2-4 weeks before last frost when soil is workable and above 45F (7C).",
        "harvest_season": "New potatoes at flowering; main crop 2-3 weeks after foliage dies back.",
        "notes": "Hill soil around stems to prevent green tubers. Rotate to reduce scab and late blight pressure.",
    },
    "soybean": {
        "planting_season": "Direct sow when soil is 60F (15C) or warmer, typically late spring after last frost.",
        "harvest_season": "Harvest when pods are dry and seeds rattle, usually late summer to early fall.",
        "notes": "Fixes nitrogen in soil. Inoculate seed in new fields. Watch for aphids, bean leaf beetle, and white mold.",
    },
    "cotton": {
        "planting_season": "Plant when soil reaches 65F (18C), often 4-6 weeks after last frost in warm climates.",
        "harvest_season": "Harvest when bolls open fully, typically late summer through fall depending on variety.",
        "notes": "Long frost-free season required. Monitor for bollworm, aphids, and timely defoliation before pick.",
    },
    "lettuce": {
        "planting_season": "Sow early spring and late summer for fall crops; prefers cool soil below 75F (24C).",
        "harvest_season": "Harvest 30-70 days after sowing; pick outer leaves or whole heads before bolting.",
        "notes": "Shade in hot weather to delay bolting. Succession plant every 2 weeks for steady supply.",
    },
    "carrot": {
        "planting_season": "Direct sow in early spring as soon as soil can be worked; late summer sowing for fall harvest.",
        "harvest_season": "Harvest 60-80 days after sowing when roots reach desired size and color.",
        "notes": "Needs loose, stone-free soil. Thin seedlings to 2-3 inches apart. Consistent moisture prevents cracking.",
    },
    "onion": {
        "planting_season": "Sets or transplants in early spring; direct seed as soon as soil is workable in cool regions.",
        "harvest_season": "Harvest when tops fall over and necks soften, usually mid to late summer.",
        "notes": "Day-length sensitive varieties matter. Cure bulbs in dry shade for 2-3 weeks before storage.",
    },
    "pepper": {
        "planting_season": "Start indoors 8-10 weeks before last frost; transplant when nights stay above 55F (13C).",
        "harvest_season": "Pick green peppers from 60-90 days; allow extra time for full color ripening.",
        "notes": "Heat-loving crop. Mulch to retain moisture. Watch for aphids, blossom end rot, and sunscald.",
    },
    "cucumber": {
        "planting_season": "Direct sow or transplant 1-2 weeks after last frost when soil is above 65F (18C).",
        "harvest_season": "Harvest 50-70 days after planting while fruit is firm and before seeds harden.",
        "notes": "Provide trellis in humid areas to improve airflow. Pick regularly to keep plants producing.",
    },
    "beans": {
        "planting_season": "Direct sow after last frost when soil is at least 60F (15C); avoid soaking wet soil.",
        "harvest_season": "Snap beans in 50-60 days; dry beans when pods are brown and brittle.",
        "notes": "Do not soak seeds before planting. Rotate with grains to break root rot cycles.",
    },
    "barley": {
        "planting_season": "Spring barley: sow as early as soil can be worked. Winter barley: sow in fall in mild climates.",
        "harvest_season": "Harvest when grain is hard and straw is golden, typically 90-110 days after planting.",
        "notes": "Tolerates cooler conditions than corn. Good cover crop option. Scout for scald and net blotch.",
    },
    "sunflower": {
        "planting_season": "Direct sow after last frost when soil is 55F (13C) or warmer, through early summer.",
        "harvest_season": "Harvest when back of head turns yellow-brown and seeds are plump, late summer to fall.",
        "notes": "Deep roots help drought tolerance. Protect from birds near maturity with mesh or early harvest.",
    },
    "spinach": {
        "planting_season": "Sow in early spring and again in late summer; germinates best below 70F (21C).",
        "harvest_season": "Harvest outer leaves from 37-45 days or cut whole plant before bolting in heat.",
        "notes": "Fast cool-season crop. Use row cover against leaf miners. Succession sow for continuous harvest.",
    },
    "broccoli": {
        "planting_season": "Transplant 2-3 weeks before last frost for spring crop; late summer for fall harvest.",
        "harvest_season": "Cut central head 60-90 days after transplant when buds are tight and green.",
        "notes": "Side shoots follow main head. Fertilize steadily. Watch for cabbage worm and aphids.",
    },
    "apple": {
        "planting_season": "Plant dormant bare-root trees in late winter or early spring before bud break.",
        "harvest_season": "Harvest varies by cultivar from late summer through fall when fruit separates easily with a lift-and-twist.",
        "notes": "Requires cross-pollination from compatible varieties. Prune annually for light penetration and fruit quality.",
    },
}


def _normalize_crop_name(crop_name: str) -> str:
    return crop_name.strip().lower().replace("-", " ").split()[0]


def _build_crop_calendar_response(crop_name: str) -> str:
    key = _normalize_crop_name(crop_name)
    if key not in CROP_CALENDAR_DATA:
        available = ", ".join(sorted(CROP_CALENDAR_DATA.keys()))
        return (
            f"No calendar entry found for '{crop_name}'. "
            f"Available crops: {available}."
        )

    data = CROP_CALENDAR_DATA[key]
    return (
        f"Crop calendar for {key.title()}:\n"
        f"- Planting season: {data['planting_season']}\n"
        f"- Harvest season: {data['harvest_season']}\n"
        f"- Notes: {data['notes']}"
    )


@tool
def crop_calendar(crop_name: str) -> str:
    """Look up planting and harvesting season information for common crops.

    Use this when farmers ask about when to plant or harvest a specific crop,
    growing seasons, or crop timing guidance.

    Args:
        crop_name: Name of the crop, such as tomato, corn, wheat, or rice.
    """
    return _build_crop_calendar_response(crop_name)


def get_weather_tool() -> OpenWeatherMapQueryRun:
    """Return an OpenWeatherMap query tool configured from env vars."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENWEATHER_API_KEY is not set. Add it to your .env file."
        )

    os.environ["OPENWEATHERMAP_API_KEY"] = api_key
    wrapper = OpenWeatherMapAPIWrapper()
    return OpenWeatherMapQueryRun(api_wrapper=wrapper)


def get_search_tool() -> DuckDuckGoSearchRun:
    """Return a DuckDuckGo search tool for web lookups."""
    return DuckDuckGoSearchRun()


def get_tools() -> list:
    """Return all agent tools: weather, search, and crop calendar."""
    return [get_weather_tool(), get_search_tool(), crop_calendar]
