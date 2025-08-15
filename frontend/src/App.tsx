import Chat from './components/Chat'
// Tailwind is already pulled in via main.tsx -> './index.css'

export default function App() {
  return (
    <main className="min-h-screen bg-white text-gray-900">
      <Chat />
    </main>
  )
}
