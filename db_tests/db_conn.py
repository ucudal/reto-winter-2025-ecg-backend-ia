import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import urllib.parse

# ====================== CONFIGURACIÓN GLOBAL ======================
MONGO_USER = "Ecg"
MONGO_PASSWORD = "Reto-2025"

MONGO_URI = 'mongodb://root:hK43CrNUq1@mongodb.reto-ucu.net:50005/?authSource=admin'

client = AsyncIOMotorClient(MONGO_URI)
db = client["admin"]
collection = db["ecg_collection"]
retrain_collection = db["retrain"]
feedback_collection = db["feedback"]

# ======================== FUNCIONES PRINCIPALES ========================

async def test_connection():
    try:
        await client.server_info()
        print("✅ MongoDB connection to Azure was successful")
    except Exception as e:
        print("❌ Error connecting to MongoDB:", str(e))

async def ensure_collection():
    try:
        existing_collections = await db.list_collection_names()
        if "ecg_collection" not in existing_collections:
            await db.create_collection("ecg_collection")
            print("✅ 'ecg_collection' created successfully")
        else:
            print("ℹ️ 'ecg_collection' already exists")
    except Exception as e:
        print("❌ Error checking/creating collection:", str(e))

async def main():
    await test_connection()
    await ensure_collection()

if __name__ == "__main__":
    asyncio.run(main())


# desde acá es codigo de copilot

# BUSCAR DATOS
async def find_ecg_by_prediction(prediction_type):
    """Busca todos los ECGs de un tipo específico"""
    try:
        cursor = collection.find({"prediction": prediction_type})
        results = await cursor.to_list(length=100)  # Máximo 100 resultados
        return results
    except Exception as e:
        print(f"Error en búsqueda: {e}")
        return []

async def find_ecg_by_confidence(min_confidence=0.8):
    """Busca ECGs con confianza mayor a un valor"""
    try:
        cursor = collection.find({"confidence": {"$gte": min_confidence}})
        results = await cursor.to_list(length=100)
        return results
    except Exception as e:
        print(f"Error en búsqueda: {e}")
        return []

async def find_one_ecg(ecg_id):
    """Busca un ECG específico por ID"""
    try:
        from bson import ObjectId
        result = await collection.find_one({"_id": ObjectId(ecg_id)})
        return result
    except Exception as e:
        print(f"Error al buscar ECG: {e}")
        return None


# ========== FUNCIONES PARA EXPLORAR LA BASE DE DATOS ==========

async def get_all_collections():
    """Obtiene la lista de todas las colecciones (equivalente a 'tablas') en la base de datos"""
    try:
        collections = await db.list_collection_names()
        print("📊 Colecciones disponibles en la base de datos:")
        for i, collection_name in enumerate(collections, 1):
            print(f"  {i}. {collection_name}")
        return collections
    except Exception as e:
        print(f"Error al obtener colecciones: {e}")
        return []

async def get_collection_stats(collection_name=None):
    """Obtiene estadísticas de una colección específica o todas"""
    try:
        if collection_name:
            # Estadísticas de una colección específica
            target_collection = db[collection_name]
            count = await target_collection.count_documents({})
            
            # Obtener un documento de ejemplo para ver la estructura
            sample_doc = await target_collection.find_one()
            
            print(f"\n📈 Estadísticas de '{collection_name}':")
            print(f"  • Total de documentos: {count}")
            if sample_doc:
                print(f"  • Campos disponibles: {list(sample_doc.keys())}")
            else:
                print(f"  • La colección está vacía")
                
            return {"collection": collection_name, "count": count, "sample": sample_doc}
        else:
            # Estadísticas de todas las colecciones
            collections = await db.list_collection_names()
            stats = {}
            
            print("📊 Estadísticas generales de la base de datos:")
            for coll_name in collections:
                coll = db[coll_name]
                count = await coll.count_documents({})
                stats[coll_name] = count
                print(f"  • {coll_name}: {count} documentos")
                
            return stats
    except Exception as e:
        print(f"Error al obtener estadísticas: {e}")
        return {}

async def explore_collection_structure(collection_name):
    """Explora la estructura de una colección mostrando ejemplos de documentos"""
    try:
        target_collection = db[collection_name]
        
        # Obtener algunos documentos de ejemplo
        cursor = target_collection.find().limit(3)
        documents = await cursor.to_list(length=3)
        
        print(f"\n🔍 Estructura de la colección '{collection_name}':")
        
        if not documents:
            print("  La colección está vacía")
            return []
            
        for i, doc in enumerate(documents, 1):
            print(f"\n  📄 Documento {i}:")
            for key, value in doc.items():
                # Mostrar tipo de dato y valor (truncado si es muy largo)
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                print(f"    {key}: {type(value).__name__} = {value_str}")
                
        return documents
    except Exception as e:
        print(f"Error al explorar colección: {e}")
        return []

async def get_database_info():
    """Obtiene información general de la base de datos"""
    try:
        # Información del servidor
        server_info = await client.server_info()
        
        # Lista de bases de datos
        db_list = await client.list_database_names()
        
        # Estadísticas de la base de datos actual
        db_stats = await db.command("dbStats")
        
        print("🗄️  INFORMACIÓN DE LA BASE DE DATOS")
        print("=" * 50)
        print(f"📡 Servidor MongoDB: {server_info.get('version', 'N/A')}")
        print(f"🎯 Base de datos actual: {db.name}")
        print(f"📂 Bases de datos disponibles: {', '.join(db_list)}")
        print(f"💾 Tamaño de la BD: {db_stats.get('dataSize', 0)} bytes")
        print(f"📊 Total de colecciones: {db_stats.get('collections', 0)}")
        
        return {
            "server_version": server_info.get('version'),
            "current_db": db.name,
            "available_dbs": db_list,
            "db_stats": db_stats
        }
    except Exception as e:
        print(f"Error al obtener información de la BD: {e}")
        return {}

async def search_documents_by_field(collection_name, field_name, field_value, limit=10):
    """Busca documentos en una colección por un campo específico"""
    try:
        target_collection = db[collection_name]
        
        # Buscar documentos que contengan el campo con el valor especificado
        cursor = target_collection.find({field_name: field_value}).limit(limit)
        results = await cursor.to_list(length=limit)
        
        print(f"\n🔎 Búsqueda en '{collection_name}' donde {field_name} = {field_value}:")
        print(f"   Encontrados: {len(results)} documentos")
        
        return results
    except Exception as e:
        print(f"Error en búsqueda: {e}")
        return []