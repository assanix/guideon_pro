import asyncio
import logging
import os
from typing import Dict, Any

import openai
import motor.motor_asyncio
from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.filters import Command
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, InputMediaPhoto, FSInputFile
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from dotenv import load_dotenv

load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["telegram_bot_db"]
users_collection = db["users"]
feedback_collection = db["feedback"]

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

# Define the keyboard layouts
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Карта🗺️"), KeyboardButton(text="РУП ШИТиИ📚")],
        [KeyboardButton(text="Где Я?🫣"), KeyboardButton(text="Найти🔍"), KeyboardButton(text="ChatGPT🤖")],
        [KeyboardButton(text="Жалобы/Предложения📥"), KeyboardButton(text="Контакты💬")]
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

chatgpt_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Назад")]
    ],
    resize_keyboard=True
)

# Define states
class BotStates(StatesGroup):
    waiting_for_feedback = State()
    waiting_for_room_number = State()
    waiting_for_find_room = State()
    waiting_for_openai_question = State()

class UserManager:
    @staticmethod
    async def add_user(user_id: int):
        await users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id}},
            upsert=True
        )

    @staticmethod
    async def save_feedback(user_id: int, feedback: str):
        await feedback_collection.insert_one({
            "user_id": user_id,
            "feedback": feedback
        })

    @staticmethod
    async def send_main_keyboard(user_id: int):
        await bot.send_message(user_id, "Выберите опцию:", reply_markup=main_keyboard)

class OpenAIHandler:
    @staticmethod
    async def ask_openai(question: str) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                max_tokens=800,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            logger.error(f"Error in OpenAI request: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

class MessageHandler:
    @staticmethod
    async def send_welcome(message: types.Message):
        await UserManager.add_user(message.from_user.id)
        first_name = message.from_user.first_name
        welcome_msg = (
            f"Привет, {first_name}! \n\n"
            f"🤖 Это бот для быстрой адаптации Перваша в стенах КБТУ. "
            f"Разработанный организацией OSIT.\n\n"
            f"🔍 Здесь вы можете:\n"
            f"- Узнать где вы или же найти нужный кабинет\n"
            f"- Рабочий учебный план ШИТиИ\n"
            f"- Юзать ChatGPT в телеграме!\n"
            f"- Оставить жалобу/предложения Деканату или OSIT\n"
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
        response_text = "🔒 Логин и пароль – Helpingstudents@kbtu.kz\n\n 📞 Контакты:\n- Офис Регистратора: 8 727 357 42 81, d.fazylova@kbtu.kz, officeregistrar@kbtu.kz\n- Библиотека: 8 727 357 42 84 (вн. 241), u.bafubaeva@kbtu.kz\n- Общежитие: 8 727 357 42 42 (вн. 601), m.shopanov@kbtu.kz, a.esimbekova@kbtu.kz\n- Оплата обучения: 8 727 357 42 58 (вн. 163, 169) a.nauruzbaeva@kbtu.kz, m.aitakyn@kbtu.kz\n- Мед. центр - medcenter@kbtu.kz\n\n🏫 Деканаты:\n- Бизнес школа: 8 727 357 42 67 (вн. 352, 358), e.mukashev@kbtu.kz, a.yerdebayeva@kbtu.kz\n- Международная школа экономики: 8 727 357 42 71 (вн. 383), a.islyami@kbtu.kz, d.bisenbaeva@kbtu.kz\n- Школа информационных технологий и инженерии: 8 727 357 42 20, fit_1course@kbtu.kz\n- Школа прикладной математики: 8 727 357 42 25, a.isakhov@kbtu.kz, n.eren@kbtu.kz\n- Школа энергетики и нефтегазовой индустрии: 8 727 357 42 42 (вн. 324), a.ismailov@kbtu.kz, a.abdukarimov@kbtu.kz\n- Школа геологии: 8 727 357 42 42 (вн. 326), a.akhmetzhanov@kbtu.kz, g.ulkhanova@kbtu.kz\n- Казахстанская морская академия: 8 727 357 42 27 (вн. 390, 392), r.biktashev@kbtu.kz, s.dlimbetova@kbtu.kz\n- Школа химической инженерии: 8 727 291 57 84, +8 727 357 42 42 (вн. 492), k.dzhamansarieva@kbtu.kz, n.saparbaeva@kbtu.kz\n- Лаборатория альтернативной энергетики и нанотехнологий: 8 727 357 42 66 (вн. 550), n.beisenkhanov@kbtu.kz, z.bugybai@kbtu.kz\n"
        await bot.send_message(message.chat.id, response_text)

    @staticmethod
    async def handle_feedback(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id,
            "Здесь вы можете анонимно написать жалобу, предложение или же сказать спасибо не только OSIT а так же Деканату ШИТиИ. Все письма будут проверяться и читаться. Удачи!",
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

        await UserManager.send_main_keyboard(message.from_user.id)

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
            reply_markup=find_keyboard,
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

        if message.text.isdigit():
            await MessageHandler.handle_room_number(message, state)
            found = True

        elif message.text.lower() in location_mapping:
            location, map_path = location_mapping[message.text.lower()]
            await bot.send_message(message.from_user.id, location)
            await bot.send_photo(message.from_user.id, FSInputFile(map_path))
            found = True

        elif message.text.lower() in room_number_mapping:
            response = room_number_mapping[message.text.lower()]
            await bot.send_message(message.from_user.id, response, reply_markup=main_keyboard)
            found = True

        if not found:
            await bot.send_message(
                message.from_user.id,
                "Номер комнаты или название не распознано. Пожалуйста, введите корректный номер или название.",
            )

        await state.clear()

    @staticmethod
    async def handle_chatgpt(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id, "Задайте свой вопрос:", reply_markup=chatgpt_keyboard
        )
        await state.set_state(BotStates.waiting_for_openai_question)

    @staticmethod
    async def process_openai_question(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id,
            "Ваш запрос принят. Пожалуйста, подождите, это займет некоторое время.",
        )
        response = await OpenAIHandler.ask_openai(message.text)
        await bot.send_message(message.from_user.id, response)
        await state.clear()

    @staticmethod
    async def handle_back(message: types.Message, state: FSMContext):
        await bot.send_message(
            message.from_user.id,
            "Вы вернулись назад.",
            reply_markup=main_keyboard,
        )
        await state.clear()


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


@router.message(F.text == "РУП ШИТиИ📚")
async def rup_command(message: types.Message):
    await MessageHandler.handle_rup(message)


@router.message(F.text.in_({"ВТИПО", "ИС", "АИУ", "РИМ", "IT management"}))
async def rup_option_command(message: types.Message):
    await MessageHandler.handle_rup_options(message)


@router.message(F.text == "Назад")
async def back_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_back(message, state)


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


@router.message(F.text == "ChatGPT🤖")
async def chatgpt_command(message: types.Message, state: FSMContext):
    await MessageHandler.handle_chatgpt(message, state)


@router.message(BotStates.waiting_for_openai_question)
async def process_openai_question_command(message: types.Message, state: FSMContext):
    await MessageHandler.process_openai_question(message, state)


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
