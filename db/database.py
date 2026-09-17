from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
DATABASE_URL = "postgresql://postgres.xppwgbnjuhrjkfdjvatq:StReAmQuAnT@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

