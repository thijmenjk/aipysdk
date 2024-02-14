'use client';

import { useChat } from 'ai/react';

export default function Page() {
  const { data, messages, input, handleInputChange, handleSubmit } = useChat();

  return (
    <div className="flex justify-center p-4">
      <div className="flex flex-col w-full max-w-md">
        {messages.map(m => (
          <div key={m.id} className="whitespace-pre-wrap p-2">
            {m.role === 'user' ? 'User: ' : 'AI: '}
            {m.content}
          </div>
        ))}

        <form className="fixed bottom-0 max-w-md bg-black" onSubmit={handleSubmit}>
          <p className="text-white">Example: Can you calculate 40+2 using the calculator and tell me how it relates to the meaning of life by searching the web and the library?</p>
          <input
            className="text-black w-full p-2 mb-8 border border-gray-300 rounded shadow-xl"
            value={input}
            placeholder="Say something..."
            onChange={handleInputChange}
          />
        </form>
      </div>
      <div className="text-white p-4 w-[500px]">
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
    </div>
  );
}
