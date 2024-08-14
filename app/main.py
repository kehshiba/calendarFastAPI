import cv2
import pandas as pd
from paddleocr import PPStructure
import json
from datetime import datetime, timedelta
import pytz
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001","http://localhost:3001"],  # Adjust this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

table_engine = PPStructure(lang='en', recovery=True, return_ocr_result_in_table=True)

def convert_time(time_str):
    if not time_str or time_str == '':
        return '16:50'
    
    parts = time_str.split('.')
    if len(parts) == 2:
        hour, minute = map(int, parts)
        if 9 <= hour <= 12:
            am_pm = 'AM'
        else:
            am_pm = 'PM'
            if hour < 12:
                hour += 12
        return f"{hour:02d}:{minute:02d}"
    return None

def process_image(img):
    result = table_engine(img)

    df = None
    for line in result:
        if line.get("type") == "table":
            html_table = line.get("res", {}).get("html")
            if html_table:
                df = pd.read_html(html_table)[0]
                break

    if df is None:
        raise ValueError("No table found in the image")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Remove columns with irrelevant data
    # Assuming you want to keep only columns with time periods and activities
    relevant_columns = [col for col in df.columns if 'Date' not in col]
    df = df[relevant_columns]

    # Fill NaN values with empty strings or some default value
    df = df.fillna('')

    # Get current week dates
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())
    week_dates = [week_start + timedelta(days=i) for i in range(5)]

    # Timezone
    tz = pytz.timezone('Asia/Kolkata')

    json_data = []

    for date, day_schedule in zip(week_dates, df.to_dict('records')):
        for time_period, activity in day_schedule.items():
            if time_period != 'Date' and pd.notna(activity):
                try:
                    # Split time periods
                    parts = time_period.split('-')
                    if len(parts) != 2:
                        continue  # Skip if not properly formatted

                    start_time, end_time = map(str.strip, parts)
                    start_time = convert_time(start_time)
                    end_time = convert_time(end_time)
                    if start_time and end_time:
                        start_datetime = datetime.strptime(f"{date} {start_time}", '%Y-%m-%d %H:%M')
                        end_datetime = datetime.strptime(f"{date} {end_time}", '%Y-%m-%d %H:%M')
                        json_data.append({
                            'title': activity,
                            'start': start_datetime.isoformat(),
                            'end': end_datetime.isoformat(),
                        })
                except ValueError as e:
                    print(f"Error processing time period '{time_period}': {e}")
                    continue

    return json_data

import traceback

@app.post("/process_image/")
async def process_image_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        json_data = process_image(img)
        return JSONResponse(content=json_data)
    except Exception as e:
        print(traceback.format_exc())  # Print the full stack trace
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)