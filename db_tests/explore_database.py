# explore_database.py - Script para explorar tu base de datos MongoDB

import asyncio
from db_conn import (
    get_all_collections,
    get_collection_stats,
    explore_collection_structure,
    get_database_info,
    search_documents_by_field
)

async def main():
    """Función principal para explorar la base de datos"""
    
    print("🚀 EXPLORANDO LA BASE DE DATOS MONGODB")
    print("=" * 60)
    
    # 1. Información general de la base de datos
    print("\n1️⃣  INFORMACIÓN GENERAL")
    await get_database_info()
    
    # 2. Obtener todas las colecciones (equivalente a "tablas")
    print("\n2️⃣  COLECCIONES DISPONIBLES")
    collections = await get_all_collections()
    
    # 3. Estadísticas de todas las colecciones
    print("\n3️⃣  ESTADÍSTICAS GENERALES")
    await get_collection_stats()
    
    # 4. Explorar estructura de cada colección
    print("\n4️⃣  ESTRUCTURA DETALLADA DE CADA COLECCIÓN")
    for collection_name in collections:
        print(f"\n{'='*20} {collection_name.upper()} {'='*20}")
        await explore_collection_structure(collection_name)
        
        # Estadísticas específicas de esta colección
        await get_collection_stats(collection_name)
    
    print("\n✅ Exploración completada!")

async def interactive_explorer():
    """Explorador interactivo de la base de datos"""
    
    print("\n🔧 EXPLORADOR INTERACTIVO")
    print("Comandos disponibles:")
    print("  1. 'collections' - Ver todas las colecciones")
    print("  2. 'stats [nombre_coleccion]' - Ver estadísticas")
    print("  3. 'explore [nombre_coleccion]' - Ver estructura")
    print("  4. 'search [coleccion] [campo] [valor]' - Buscar documentos")
    print("  5. 'info' - Información de la base de datos")
    print("  6. 'quit' - Salir")
    
    while True:
        try:
            command = input("\n💻 Ingresa comando: ").strip().lower()
            
            if command == 'quit':
                print("👋 ¡Hasta luego!")
                break
            elif command == 'collections':
                await get_all_collections()
            elif command == 'info':
                await get_database_info()
            elif command.startswith('stats'):
                parts = command.split()
                if len(parts) == 1:
                    await get_collection_stats()
                else:
                    await get_collection_stats(parts[1])
            elif command.startswith('explore'):
                parts = command.split()
                if len(parts) > 1:
                    await explore_collection_structure(parts[1])
                else:
                    print("❌ Uso: explore [nombre_coleccion]")
            elif command.startswith('search'):
                parts = command.split()
                if len(parts) >= 4:
                    collection_name = parts[1]
                    field_name = parts[2]
                    field_value = parts[3]
                    results = await search_documents_by_field(
                        collection_name, field_name, field_value
                    )
                    for i, doc in enumerate(results, 1):
                        print(f"\n📄 Resultado {i}: {doc}")
                else:
                    print("❌ Uso: search [coleccion] [campo] [valor]")
            else:
                print("❌ Comando no reconocido")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Función específica para ver qué hay en tu base de datos actual
async def quick_overview():
    """Vista rápida de lo que tienes en la base de datos"""
    
    print("⚡ VISTA RÁPIDA DE TU BASE DE DATOS")
    print("=" * 40)
    
    # Ver colecciones
    collections = await get_all_collections()
    
    if not collections:
        print("\n📭 Tu base de datos está vacía (no hay colecciones)")
        return
    
    # Ver contenido de las colecciones principales
    main_collections = ['ecg_collection', 'retrain', 'feedback']
    
    for coll_name in main_collections:
        if coll_name in collections:
            print(f"\n🔍 Revisando '{coll_name}'...")
            await get_collection_stats(coll_name)
            
            # Mostrar un documento de ejemplo si existe
            from db_conn import db
            target_collection = db[coll_name]
            sample = await target_collection.find_one()
            if sample:
                print(f"   📄 Ejemplo de documento:")
                for key, value in list(sample.items())[:5]:  # Primeros 5 campos
                    print(f"     {key}: {value}")
                if len(sample) > 5:
                    print(f"     ... y {len(sample)-5} campos más")

if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1. Vista rápida")
    print("2. Exploración completa") 
    print("3. Explorador interactivo")
    
    choice = input("Opción (1/2/3): ").strip()
    
    if choice == "1":
        asyncio.run(quick_overview())
    elif choice == "2":
        asyncio.run(main())
    elif choice == "3":
        asyncio.run(interactive_explorer())
    else:
        print("Opción inválida")
