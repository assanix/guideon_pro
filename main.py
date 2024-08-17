import asyncio
import logging
import os
import re
import hashlib
from typing import Dict, Any, List

import openai
import google.generativeai as genai
import motor.motor_asyncio
import tiktoken
import docx
import PyPDF2
import openpyxl
from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.filters import Command
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, InputMediaPhoto, FSInputFile
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from dotenv import load_dotenv
from pinecone import Pinecone
import aiofiles

load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "kbtu-docs"
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)


# Initialize MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["telegram_bot_db"]
users_collection = db["users"]
feedback_collection = db["feedback"]

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

# Admins ids
ADMINS = [1138549375]

# Define the keyboard layouts
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🗺️ Карта"), KeyboardButton(text="📚 РУП ШИТиИ")],
        [KeyboardButton(text="🫣 Где Я?"), KeyboardButton(text="🔍 Найти"), KeyboardButton(text="🤖 MentorGPT")],
        [KeyboardButton(text="📥 Жалобы/Предложения"), KeyboardButton(text="💬 Контакты")]
    ],
    resize_keyboard=True
)

chatgpt_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="⬅️ Назад")]
    ],
    resize_keyboard=True
)

find_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Кушать"), KeyboardButton(text="Учиться")],
        [KeyboardButton(text="Назад")]
    ],
    resize_keyboard=True
)

class UnicodeFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.encode('utf-8').decode('unicode_escape')
        return super(UnicodeFormatter, self).format(record)

for handler in logger.handlers:
    handler.setFormatter(UnicodeFormatter())

async def clear_pinecone_index():
    try:
        index.delete(delete_all=True)
        logger.debug("All vectors deleted from Pinecone index.")
    except Exception as e:
        logger.error(f"Error deleting vectors from Pinecone: {e}")
        raise

async def reprocess_uploads():
    upload_dir = "./uploads/"
    files = os.listdir(upload_dir)
    for file_name in files:
        file_path = os.path.join(upload_dir, file_name)
        await FileHandler.process_and_store_document(file_path, file_name, 0)

async def handle_main_menu_button(message: types.Message, state: FSMContext):
    await state.clear()
    if message.text == "🗺️ Карта":
        await MessageHandler.handle_map(message)
    elif message.text == "📚 РУП ШИТиИ":
        await MessageHandler.handle_rup(message)
    elif message.text == "🫣 Где Я?":
        await MessageHandler.ask_for_room_number(message, state)
    elif message.text == "🔍 Найти":
        await MessageHandler.handle_find_room(message, state)
    elif message.text == "📥 Жалобы/Предложения":
        await MessageHandler.handle_feedback(message, state)
    elif message.text == "💬 Контакты":
        await MessageHandler.handle_contacts(message)



# Define states
class BotStates(StatesGroup):
    waiting_for_feedback = State()
    waiting_for_room_number = State()
    waiting_for_find_room = State()
    asking_question = State()

# Helper functions for Pinecone integration
async def get_vector_from_text(text: str) -> List[float]:
    try:
        if not text or len(text) < 3:  # Check if the text is too short
            raise ValueError("Text is empty or too short to generate a meaningful embedding.")

        # Tokenize and chunk the text
        tokenizer = tiktoken.encoding_for_model(model_name="text-embedding-ada-002", )
        tokens = tokenizer.encode(text)

        max_tokens = 8192
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

        vectors = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk)
            response = await openai.Embedding.acreate(
                input=chunk_text,
                model="text-embedding-ada-002"
            )
            embedding = response["data"][0]["embedding"]

            if any(map(lambda x: x != x, embedding)):  # Check for NaN
                logger.error("Generated embedding contains NaN values.")
                continue

            vectors.append(embedding)

        if not vectors:
            raise ValueError("No valid embeddings were generated.")

        averaged_vector = [sum(x)/len(x) for x in zip(*vectors)]
        return averaged_vector

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise ValueError("No valid embeddings were generated.")


async def add_vectors_to_pinecone(vectors: List[Dict[str, Any]]):
    try:
        for vector in vectors:
            # Truncate the metadata if it exceeds the limit
            if len(vector['metadata']['text']) > 40960:
                vector['metadata']['text'] = vector['metadata']['text'][:40000] + '...'

        response = index.upsert(vectors)
    except Exception as e:
        logger.error(f"Error adding vectors to Pinecone: {e}")
        raise



async def query_pinecone(vector: List[float]) -> str:
    results = index.query(vector=vector, top_k=100, include_metadata=True)
    contexts = [match["metadata"]["text"] for match in results["matches"]]

    return "\n\n".join(contexts) if contexts else "Извините, я не смог найти подходящий ответ."

async def generate_answer_from_context(question: str, context: str) -> str:
    try:
#         model = genai.GenerativeModel(
#             model_name="gemini-1.5-flash",
#             system_instruction=""""You are a friendly, empathetic, and knowledgeable mentor's assistant at KBTU, guiding first-year students. Your primary role is to provide clear, structured, and supportive answers based on the documents and information provided.
#
# 1. **Focus equally on academic guidance, technical support, and emotional assistance** to ensure the student feels fully supported.
# 2. If a question touches on a sensitive topic, **respond with empathy** and suggest seeking help from a professional like a psychologist or mentor if needed.
# 3. If you cannot find the relevant information in the documents, **acknowledge this** and gently recommend that the student reach out to their mentor for further assistance.
# 4. Keep the **tone casual, empathetic, and supportive**, avoiding overly formal language. Use simple language and **emojis** where appropriate to make the conversation engaging and friendly.
#
# **Handling Different Terms:**
# - Sometimes, students might use terms in Cyrillic that are meant to represent concepts commonly written in Latin script (e.g., 'адд дроп' for 'Add/drop'). Recognize these terms and respond accordingly by matching them to their Latin equivalents when possible.
#
# Example Interaction:
# - Student: 'Как мне оплатить ретейк?'
# - Bot: 'Чтобы оплатить ретейк, тебе нужно... 😊'
#
# If you encounter a term like 'адд дроп', recognize it as 'Add/drop' and provide the appropriate guidance.
#
# Always strive to make the student feel supported, understood, and encouraged."
# """
#         )
#         response = model.generate_content(f"Вопрос: {question}\nКонтекст: {context}")
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
You are a friendly, empathetic, and knowledgeable mentor's assistant at School of Information technology and Engineering, also known as "Site", "ШИТиИ", "ФИТ" at KBTU, guiding first-year students. Your primary role is to provide clear, structured, and supportive answers based on the documents and information provided without using asterisks in text.
Your goal - helps for mentors and in cases where you are not sure of the answer, refer the student to your mentor.
1. Focus equally on academic guidance, technical support, and emotional assistance to ensure the student feels fully supported.
2. If a question touches on a sensitive topic, respond with empathy and suggest seeking help from a professional like a psychologist or mentor if needed.
3. If you cannot find the relevant information in the documents, acknowledge this and gently recommend that the student reach out to their mentor for further assistance.
4. Keep the tone casual, empathetic, and supportive, avoiding overly formal language. Use simple language and emojis where appropriate to make the conversation engaging and friendly.
5. If a student asks for help in choosing elective courses (also referred to as "элективки" or "мажорки"), provide them with a list of 6 elective courses on english with their codes that best match their request and describe reason, based on the information stored in the database. Always ensure these recommendations are directly sourced from the available documents.

Handling Different Terms:
- Sometimes, students might use terms in Cyrillic that are meant to represent concepts commonly written in Latin script (e.g., 'адд дроп' for 'Add/drop'). Recognize these terms and respond accordingly by matching them to their Latin equivalents when possible.

Example Interaction:
- Student: 'Как мне оплатить ретейк?'
- Bot: 'Чтобы оплатить ретейк, тебе нужно... 😊'

If you encounter a term like 'адд дроп', recognize it as 'Add/drop' and provide the appropriate guidance. Or if you encounter a term like "ор", recognize it as "Офис Регистратора" and provide the appropriate guidance.

Always strive to make the student feel supported, understood, and encouraged.
"""},
                {"role": "user", "content": f"Вопрос: {question}\nКонтекст: {context}"},
            ],
        )
        # response_text = response.text.strip()
        response_text = response.choices[0].message["content"].strip()
        response_text = re.sub(r'#\s*', '', response_text)
        # logger.debug(f"\n\n========CONTEXT===========\n\n{context}")
        return response_text
    except Exception as e:
        logger.error(f"Error in OpenAI request: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."


class FileHandler:

    UPLOAD_DIR = "./uploads/"

    @staticmethod
    def generate_ascii_id(file_name: str) -> str:
        hash_object = hashlib.sha256(file_name.encode('utf-8'))
        return hash_object.hexdigest()

    @staticmethod
    async def handle_file_upload(message: types.Message):
        if not await UserManager.is_admin(message.from_user.id):
            await bot.send_message(message.from_user.id, "У вас нет прав для загрузки файлов.")
            return

        documents = message.document if isinstance(message.document, list) else [message.document]
        uploaded_files = []
        for document in documents:
            file = await bot.get_file(document.file_id)
            file_path = os.path.join(FileHandler.UPLOAD_DIR, document.file_name)
            await bot.download_file(file.file_path, file_path)

            # Process and store the document
            await FileHandler.process_and_store_document(file_path, document.file_name, message.from_user.id)
            uploaded_files.append(document.file_name)

        await bot.send_message(message.from_user.id, f"Все файлы успешно загружены и обработаны:\n" + "\n".join(uploaded_files))

    @staticmethod
    async def process_and_store_document(file_path: str, file_name: str, user_id: int):
        text = None
        if file_path.endswith('.docx'):
            text = FileHandler.extract_text_from_docx(file_path)
        elif file_path.endswith('.pdf'):
            text = FileHandler.extract_text_from_pdf(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            text = FileHandler.extract_text_from_excel(file_path)
        else:
            try:
                async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
                    text = await file.read()
            except UnicodeDecodeError:
                try:
                    async with aiofiles.open(file_path, mode='r', encoding='cp1251') as file:
                        text = await file.read()
                except UnicodeDecodeError:
                    async with aiofiles.open(file_path, mode='rb') as file:
                        text = await file.read()

        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')

        text = text.strip()

        if not text:
            logger.error("Text is empty or invalid after reading from file.")
            await bot.send_message(user_id, "Ошибка: текст в файле пуст или некорректен.")
            return

        # Split the text into smaller chunks
        chunk_size = 4000  # Adjust this size based on your needs
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        try:
            for i, chunk in enumerate(text_chunks):
                vector = await get_vector_from_text(chunk)
                chunk_id = f"{FileHandler.generate_ascii_id(file_name)}_{i}"
                vectors = [
                    {"id": chunk_id, "values": vector, "metadata": {"text": chunk[:2000], "filename": file_name}}]


                await add_vectors_to_pinecone(vectors)

        except ValueError as ve:
            logger.error(f"Failed to process document: {ve}")
            await bot.send_message(user_id, "Ошибка: не удалось обработать файл.")
        except Exception as e:
            logger.error(f"Unexpected error processing document: {e}")
            await bot.send_message(user_id, "Ошибка: не удалось обработать файл.")

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    @staticmethod
    def  extract_text_from_pdf(file_path: str) -> str:
        full_text = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                full_text.append(page.extract_text())
        return '\n'.join(full_text)

    @staticmethod
    def extract_text_from_excel(file_path: str) -> str:
        wb = openpyxl.load_workbook(file_path)
        full_text = []
        for sheet in wb:
            for row in sheet.iter_rows(values_only=True):
                row_text = " ".join([str(cell) for cell in row if cell is not None])
                full_text.append(row_text)
        return '\n'.join(full_text)

    @staticmethod
    async def delete_vectors_by_filename(filename: str):
        try:
            vector_id = FileHandler.generate_ascii_id(filename)
            index.delete(ids=[vector_id])
            logger.debug(f"Deleted vector with ID: {vector_id}")
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            raise

    @staticmethod
    async def delete_file(filename: str):
        file_path = os.path.join(FileHandler.UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Deleted file: {file_path}")
        else:
            logger.error(f"File not found: {file_path}")

    @staticmethod
    async def handle_delete_request(message: types.Message, filename: str):
        await FileHandler.delete_vectors_by_filename(filename)
        await FileHandler.delete_file(filename)
        await bot.send_message(message.from_user.id, f"Файл '{filename}' и его векторные данные успешно удалены.")


class MessageHandler:
    @staticmethod
    async def send_welcome(message: types.Message):
        await UserManager.add_user(
            user_id=message.from_user.id,
            username=message.from_user.username,
            first_name=message.from_user.first_name
        )
        first_name = message.from_user.first_name or "Пользователь"
        welcome_msg = (
            f"Привет, {first_name}! \n\n"
            f"🤖 Это бот для быстрой адаптации Перваша в стенах КБТУ. "
            f"Разработанный организацией занимающейся менторской программой.\n\n"
            f"🔍 Здесь вы можете:\n"
            f"- Узнать где вы или же найти нужный кабинет\n"
            f"- Рабочий учебный план ШИТиИ\n"
            f"- Задать вопрос на основе данных, которые мы уже загрузили в систему.\n"
            f"- Оставить жалобу/предложения Деканату или менторской программе\n"
        )
        await bot.send_message(message.from_user.id, welcome_msg, reply_markup=main_keyboard)

    @staticmethod
    async def handle_map(message: types.Message):
        maps_paths = [
            "./map/1-floor_kbtu.jpg",
            "./map/2-floor_kbtu.jpg",
            "./map/3-floor_kbtu.jpg",
            "./map/4-floor_kbtu.jpg",
            "./map/5-floor_kbtu.jpg",
        ]
        media = [InputMediaPhoto(media=FSInputFile(map_path)) for map_path in maps_paths]

        await bot.send_media_group(message.from_user.id, media)

    @staticmethod
    async def handle_contacts(message: types.Message):
        response_text = "Соц сети: \n\n №1 Технический вуз Казахстана (@kbtu_official) • Фото и видео в InstagramTelegram: Contact @kbtuadmission2024(23) Публикация | LinkedInKBTU (@kbtu_official) | TikTok \n\n Логин и пароль – Helpingstudents@kbtu.kz\n\n 📞 Контакты:\n- Офис Регистратора: 8 727 357 42 81, d.fazylova@kbtu.kz, officeregistrar@kbtu.kz\n- Библиотека: 8 727 357 42 84 (вн. 241), u.bafubaeva@kbtu.kz\n- Общежитие: 8 727 357 42 42 (вн. 601), m.shopanov@kbtu.kz, a.esimbekova@kbtu.kz\n- Оплата обучения: 8 727 357 42 58 (вн. 163, 169) a.nauruzbaeva@kbtu.kz, m.aitakyn@kbtu.kz\n- Мед. центр - medcenter@kbtu.kz\n\n🏫 Деканаты:\n- Бизнес школа: 8 727 357 42 67 (вн. 352, 358), e.mukashev@kbtu.kz, a.yerdebayeva@kbtu.kz\n- Международная школа экономики: 8 727 357 42 71 (вн. 383), a.islyami@kbtu.kz, d.bisenbaeva@kbtu.kz\n- Школа информационных технологий и инженерии: 8 727 357 42 20, fit_1course@kbtu.kz\n- Школа прикладной математики: 8 727 357 42 25, a.isakhov@kbtu.kz, n.eren@kbtu.kz\n- Школа энергетики и нефтегазовой индустрии: 8 727 357 42 42 (вн. 324), a.ismailov@kbtu.kz, a.abdukarimov@kbtu.kz\n- Школа геологии: 8 727 357 42 42 (вн. 326), a.akhmetzhanov@kbtu.kz, g.ulkhanova@kbtu.kz\n- Казахстанская морская академия: 8 727 357 42 27 (вн. 390, 392), r.biktashev@kbtu.kz, s.dlimbetova@kbtu.kz\n- Школа химической инженерии: 8 727 291 57 84, +8 727 357 42 42 (вн. 492), k.dzhamansarieva@kbtu.kz, n.saparbaeva@kbtu.kz\n- Лаборатория альтернативной энергетики и нанотехнологий: 8 727 357 42 66 (вн. 550), n.beisenkhanov@kbtu.kz, z.bugybai@kbtu.kz\n"
        await bot.send_message(message.chat.id, response_text)

    @staticmethod
    async def handle_feedback(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id,
            "Здесь вы можете анонимно написать жалобу, предложение или же сказать спасибо не только Менторской программе, а так же Деканату ШИТиИ. Все письма будут проверяться и читаться. Удачи!",
        )
        await state.set_state(BotStates.waiting_for_feedback)

    @staticmethod
    async def process_feedback(message: types.Message, state: FSMContext):
        await UserManager.save_feedback(message.from_user.id, message.text)
        await bot.send_message(message.from_user.id, "Спасибо за ваше сообщение!")
        await state.clear()

    @staticmethod
    async def handle_rup(message: types.Message):
        rup_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="ВТИПО"), KeyboardButton(text="ИС")],
                [KeyboardButton(text="АИУ"), KeyboardButton(text="РИМ"), KeyboardButton(text="IT management")],
                [KeyboardButton(text="Назад")]
            ],
            resize_keyboard=True
        )

        await bot.send_message(
            message.from_user.id, "Выберите нужный вам РУП:", reply_markup=rup_keyboard
        )

    @staticmethod
    async def handle_rup_options(message: types.Message):
        file_paths = {
            "ВТИПО": "./rup_fit/VTIPO.pdf",
            "ИС": "./rup_fit/IS.pdf",
            "РИМ": "./rup_fit/RIM.pdf",
            "АИУ": "./rup_fit/AU.pdf",
            "IT management": "./rup_fit/it_man.pdf"
        }
        file_path = file_paths.get(message.text)
        if file_path:
            await bot.send_document(message.from_user.id, FSInputFile(file_path))



    @staticmethod
    async def handle_back(message: types.Message):
        await UserManager.send_main_keyboard(message.from_user.id)

    @staticmethod
    async def ask_for_room_number(message: types.Message, state: FSMContext):
        await bot.send_message(message.from_user.id, "Введите номер кабинета рядом с вами.")
        await state.set_state(BotStates.waiting_for_room_number)

    @staticmethod
    async def handle_room_number(message: types.Message, state: FSMContext):
        found = False
        room_mapping = {
            range(100, 144): ("Вы на Панфилова, 1 этаж.", "./map/floor1/PF.png"),
            range(144, 153): ("Вы на Толе Би, 1 этаж.", "./map/floor1/TB.png"),
            range(156, 184): ("Вы на Абылайхана, 1 этаж.", "./map/floor1/Abl.png"),
            range(252, 285): ("Вы на Абылайхана, 2 этаж.", "./map/floor2/ABL.png"),
            range(202, 246): ("Вы на Панфилова, 2 этаж.", "./map/floor2/PF.png"),
            range(246, 252): ("Вы на Толе Би, 2 этаж.", "./map/floor2/TB.png"),
            range(300, 344): ("Вы на Панфилова, 3 этаж.", "./map/floor3/PF.png"),
            range(344, 361): ("Вы на Толе Би, 3 этаж.", "./map/floor3/TB.png"),
            range(361, 389): ("Вы на Абылайхана, 3 этаж.", "./map/floor3/ABL.png"),
            range(501, 523): ("Вы на Толе Би, 5 этаж.", "./map/5floor.png"),
            range(400, 417): ("Вы на Панфилова, 4 этаж.", "./map/floor4/PF.png"),
            range(419, 439): ("Вы на Казыбек Би, 4 этаж.", "./map/floor4/KB.png"),
            range(444, 462): ("Вы на Абылайхана, 4 этаж.", "./map/floor4/ABL.png"),
            range(462, 477): ("Вы на Толе Би, 4 этаж.", "./map/floor4/TB.png"),
        }

        if message.text.isdigit():
            room_number = int(message.text)
            for room_range, (location, map_path) in room_mapping.items():
                if room_number in room_range:
                    await bot.send_message(message.from_user.id, location)
                    await bot.send_photo(message.from_user.id, FSInputFile(map_path))
                    found = True
                    break

        if not found:
            await bot.send_message(
                message.from_user.id,
                "Номер комнаты или название не распознано. Пожалуйста, введите корректный номер или название.",
            )

        await state.clear()

    @staticmethod
    async def handle_find_room(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id,
            "Введите номер или название кабинета, который вы хотите найти.\n\nМожно узнать места где рядом можно покушать или же поучиться",
        )
        await state.set_state(BotStates.waiting_for_find_room)

    @staticmethod
    async def process_find_room(message: types.Message, state: FSMContext):
        found = False
        location_mapping = {
            "халык": ("Это находиться на Казыбек би, 1 этаж.", "./map/floor1/KB.png"),
            "геймдев": ("Это находиться на Казыбек би, 2 этаже.", "./map/floor2/KB.png"),
            "коворкинг на 2 этаже": ("Это находиться на Казыбек би, 2 этаже.", "./map/floor2/KB.png"),
            "game dev": ("Это находиться на Казыбек би, 2 этаже.", "./map/floor2/KB.png"),
            "gamedev": ("Это находиться на Казыбек би, 2 этаже.", "./map/floor2/KB.png"),
            "726": ("Это находиться на Казыбек би, 2 этаже.", "./map/floor2/KB.png"),
            "столовка": ("Столовка находится на 0 этаже Толе Би.", "./map/floor1/Canteen.png"),
        }

        room_number_mapping = {
            "кушать": "Столовка находится на 0 этаже Толе Би. Купить перекус на 0, 1, 3 этаже Толе Би, а также на 1 этаже Абылайхана. Еще рядом с универом есть много заведений, где можно покушать.",
            "учиться": "Пока я не знаю, где можно учиться",
        }

        if message.text.lower() in room_number_mapping:
            response = room_number_mapping[message.text.lower()]
            await bot.send_message(message.from_user.id, response, reply_markup=main_keyboard)
            found = True
        elif message.text.isdigit():
            await MessageHandler.handle_room_number(message, state)
            found = True
        elif message.text.lower() in location_mapping:
            location, map_path = location_mapping[message.text.lower()]
            await bot.send_message(message.from_user.id, location)
            await bot.send_photo(message.from_user.id, FSInputFile(map_path))
            found = True

        if not found:
            await bot.send_message(
                message.from_user.id,
                "Номер комнаты или название не распознано. Пожалуйста, введите корректный номер или название.",
            )


    @staticmethod
    async def handle_langchain_question(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id, "Введите ваш вопрос:", reply_markup=chatgpt_keyboard
        )
        if state:
            await state.set_state(BotStates.asking_question)

    @staticmethod
    async def process_langchain_question(message: types.Message, state: FSMContext):
        if message.text in {"🗺️ Карта", "📚 РУП ШИТиИ", "🫣 Где Я?", "🔍 Найти", "📥 Жалобы/Предложения", "💬 Контакты"}:
            await handle_main_menu_button(message, state)
            return

        if message.text == "⬅️ Назад":
            await state.clear()
            await UserManager.send_main_keyboard(message.from_user.id)
            return

        await bot.send_message(
            message.from_user.id,
            "Ваш запрос принят. Пожалуйста, подождите, это займет некоторое время.",
        )

        # Get the question
        question = message.text



        try:
            query_vector = await get_vector_from_text(question)
        except ValueError as ve:
            logger.error(f"Error generating embedding: {ve}")
            await bot.send_message(message.from_user.id, "Извините, я не смог найти подходящий ответ.")
            return


        try:
            context = await query_pinecone(query_vector)
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            await bot.send_message(message.from_user.id, "Извините, произошла ошибка при обработке вашего запроса.")
            return

        # Generate an answer based on the context
        response = await generate_answer_from_context(question, context)
        response = response.replace('*', '')

        await bot.send_message(message.from_user.id, response)

    @staticmethod
    async def handle_unhandled_message(message: types.Message, state: FSMContext):
        if message.document:
            await FileHandler.handle_file_upload(message)
        else:
            if state is None or state.get_state() is None:
                # If no state is set, assume the user is asking a question
                await state.set_state(BotStates.asking_question)
                await MessageHandler.handle_langchain_question(message, state)
            else:
                await MessageHandler.process_langchain_question(message, state)




    @staticmethod
    async def handle_back(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id,
            "Вы вернулись назад.",
            reply_markup=main_keyboard,
        )
        await state.clear()

class UserManager:
    @staticmethod
    async def add_user(user_id: int, username: str = None, first_name: str = None):
        user_name = username or first_name or f"user_{user_id}"
        await users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "user_name": user_name}},
            upsert=True
        )

    @staticmethod
    async def is_admin(user_id: int) -> bool:
        return user_id in ADMINS

    @staticmethod
    async def save_feedback(user_id: int, feedback: str):
        await feedback_collection.insert_one({
            "user_id": user_id,
            "feedback": feedback
        })

    @staticmethod
    async def send_main_keyboard(user_id: int):
        await bot.send_message(user_id, "Выберите опцию:", reply_markup=main_keyboard)

# Register handlers
@router.message(Command("start"))
async def start_command(message: types.Message):
    await MessageHandler.send_welcome(message)

@router.message(F.text == "Карта🗺️")
async def map_command(message: types.Message):
    await MessageHandler.handle_map(message)


@router.message(F.text == "Контакты💬")
async def contacts_command(message: types.Message):
    await MessageHandler.handle_contacts(message)


@router.message(F.text == "Жалобы/Предложения📥")
async def feedback_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_feedback(message, state)

@router.message(BotStates.waiting_for_feedback)
async def process_feedback(message: types.Message, state: FSMContext):
    await MessageHandler.process_feedback(message, state)


@router.message(F.text == "РУП ШИТиИ📚")
async def rup_command(message: types.Message):
    await MessageHandler.handle_rup(message)


@router.message(F.text.in_({"ВТИПО", "ИС", "АИУ", "РИМ", "IT management"}))
async def rup_option_command(message: types.Message):
    await MessageHandler.handle_rup_options(message)


@router.message(F.text == "Назад")
async def back_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_back(message, state)

@router.message(F.text.in_({"🗺️ Карта", "📚 РУП ШИТиИ", "🫣 Где Я?", "🔍 Найти", "📥 Жалобы/Предложения", "💬 Контакты"}))
async def main_menu_button_command(message: types.Message, state: FSMContext):
    await handle_main_menu_button(message, state)



@router.message(F.text == "Где Я?🫣")
async def where_am_i_command(message: types.Message, state: FSMContext):
    await MessageHandler.ask_for_room_number(message, state)


@router.message(BotStates.waiting_for_room_number)
async def process_room_number_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_room_number(message, state)


@router.message(F.text == "Найти🔍")
async def find_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_find_room(message, state)


@router.message(BotStates.waiting_for_find_room)
async def process_find_room_command(message: types.Message, state: FSMContext):
    await MessageHandler.process_find_room(message, state)

@router.message(F.text == "🤖 MentorGPT")
async def langchain_question_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_langchain_question(message, state)

@router.message(BotStates.asking_question)
async def process_langchain_question_command(message: types.Message, state: FSMContext):
    await MessageHandler.process_langchain_question(message, state)

@router.message()
async def handle_unhandled_message(message: types.Message, state: FSMContext):
    await MessageHandler.handle_unhandled_message(message, state)

async def main():
    dp.include_router(router)
    # await clear_pinecone_index()
    # await reprocess_uploads()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
