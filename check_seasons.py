from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()
eng = create_engine(os.environ["DATABASE_URL"])
with eng.connect() as c:
    r = c.execute(text("SELECT DISTINCT season FROM player_career_stats ORDER BY season"))
    print("Seasons:", [row[0] for row in r])
    r = c.execute(text(
        "SELECT COUNT(*) FROM player_career_stats "
        "WHERE season=2024 AND phase='death' AND role='bowler' AND innings>=5"
    ))
    print("2024 death bowlers innings>=5:", r.scalar())
