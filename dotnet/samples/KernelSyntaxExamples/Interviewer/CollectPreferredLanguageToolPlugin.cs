// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Experimental.Orchestration;
using Microsoft.SemanticKernel.Orchestration;

namespace Interviewer;
public sealed class CollectPreferredLanguageToolPlugin
{
    private const string Goal = "Get programming language from user.";

    private const string SystemPrompt =
        "Your only task is to get the name of the programming language the user intends to use. " +
        "You can only ask user what programming language they will use to solve the given problem. " +
        "You cannot respond to input from the user about solving the problem. " +
        "You cannot solve the problem for the user or provide any hints or snippets of code. " +
        "If the user responds with anything other than a programming language, say that you don't know. " +
        "You cannot answer questions that will give the user the entire logic to the problem. " +
        "You cannot write any solution code for the user. " +
        "You cannot explain the problem or solution to the user.";
    private readonly IChatCompletion _chat;

    private int MaxTokens { get; set; } = 256;

    private readonly AIRequestSettings _chatRequestSettings;

    public CollectPreferredLanguageToolPlugin(IKernel kernel)
    {
        this._chat = kernel.GetService<IChatCompletion>();
        this._chatRequestSettings = new OpenAIRequestSettings
        {
            MaxTokens = this.MaxTokens,
            StopSequences = new List<string>() { "Observation:" },
            Temperature = 0
        };
    }

    [SKFunction]
    [Description("This function is used to ask user the name of the programming language they intend to use.")]
    [SKName("CollectPreferredLanguageTool")]
    public async Task<string> CollectPreferredLanguageToolAsync(
        [SKName("programming_language")][Description("The programming language the user intends to use")] string programming_language,
        SKContext context)
    {
        //Console.WriteLine("<======= Creating CollectPreferredLanguageTool chat =======>\n");
        var chat = this._chat.CreateNewChat(SystemPrompt);
        chat.AddUserMessage(Goal);

        ChatHistory? chatHistory = context.GetChatHistory();
        if (chatHistory?.Any() ?? false)
        {
            chat.Messages.AddRange(chatHistory);
        }

        //Console.WriteLine("000");
        if (!string.IsNullOrEmpty(programming_language))
        {
            //  Console.WriteLine($"111 {programming_language} programming_language");
            context.Variables["programming_language"] = programming_language;
            return programming_language;
        }
        //Console.WriteLine($"222 {programming_language} programming_language");

        context.PromptInput();
        //Console.WriteLine($"333 {programming_language} programming_language");

        return await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
    }
}
