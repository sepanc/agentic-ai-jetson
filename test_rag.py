"""
Quick test for RAG Agent
"""

from rag_agent import RAGAgent
import os


def main():
    print("\n" + "=" * 60)
    print("🧪 RAG Agent Test Suite")
    print("=" * 60)
    
    os.makedirs("logs", exist_ok=True)
    
    # Test 1: Initialize
    print("\n🧪 Test 1: Initialization")
    try:
        agent = RAGAgent()
        print("   ✅ Agent initialized")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 2: Load documents
    print("\n🧪 Test 2: Document loading")
    docs = agent.load_documents()
    
    if len(docs) == 0:
        print("   ⚠️  No documents found - add PDFs to ./documents/")
        return
    else:
        print(f"   ✅ Loaded {len(docs)} pages")
    
    # Test 3: Vector store
    print("\n🧪 Test 3: Vector store creation")
    try:
        agent.create_vectorstore(docs)
        print("   ✅ Vector store ready")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 4: QA chain
    print("\n🧪 Test 4: QA chain setup")
    try:
        agent.setup_qa_chain()
        print("   ✅ QA chain ready")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 5: Query
    print("\n🧪 Test 5: Sample query")
    try:
        result = agent.query("What is this document about?")
        print(f"   ✅ Answer: {result['answer'][:80]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()